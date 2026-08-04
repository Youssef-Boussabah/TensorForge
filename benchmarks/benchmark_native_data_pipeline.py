"""Characterization of the deterministic native data pipeline (Phase J, J8).

Phase J shipped a dataset (J1), a batch planner (J2), a loader with a
five-phase batch handoff (J3), loader state (J4), the caller-managed
checkpoint workflow (J5), a training example (J6), and an adversarial
hardening matrix (J7). This harness answers the one question none of them
asked: **what does each layer of that pipeline cost on this machine?**

Four separate questions, deliberately not blurred into one
--------------------------------------------------------------

1. **Dataset indexing** — what does gathering rows out of the immutable
   host snapshot cost, and what does `target_batch` cost?
2. **Batch planning** — what does `plan()` / `next_batch_indices()` cost?
3. **Permutation construction** — what does one epoch's deterministic
   shuffled order cost, cold and on a cache hit?
4. **Host-to-native materialization** — what does
   `dataset.feature_batch(indices)` cost, allocation and transfer
   included?

A fifth family, `loader_delivery`, measures one whole `next(iterator)` —
the §9.4 transaction end to end. It is a **composition**, reported
separately and never as a substitute for the four isolated layers: a
single end-to-end number cannot say which layer dominates, and §22.3 of
`docs/native_data_pipeline_design.md` makes "which layer dominates" the
only question that could ever reopen the export count. Answering it needs
the layers apart.

What this harness is not
------------------------

**J8 is characterization only, and ships no optimization.** Nothing here
changes runtime behavior, and no measurement below may be used to change
one without its own separately approved milestone.

**Nothing asserts a speed.** No timing threshold, no duration budget, no
throughput floor, no ratio limit, no comparison against a stored number,
and **no CI job that fails on one**. There is **no result file of any
kind** — not JSON, not CSV, not a cache, not a baseline. `--json` writes
to *stdout* and nowhere else, and the CLI has no output-path option
because it has nothing to write.

**Correctness runs before timing, always.** Every case validates its
result against an independent oracle *before* the timing helper is
reached, so a failed gate publishes no timing and the CLI exits nonzero
with clean stdout. The gates are exact — index tuples, plans, and
permutations by equality; feature values by raw IEEE-754 bit patterns
inside one dtype; targets by exact `int64` equality; dtype, shape,
device, ownership, contiguity, and the read-only flag by identity. No
`allclose`, no tolerance, and no approximation appears anywhere: every
operation this pipeline performs is a copy, a gather, or integer
planning, and for those a tolerance would be asserting less than the
contract promises.

**The two dtypes are never divided by one another.** float32 and float64
are measured separately and reported separately. A float32/float64 speed
ratio is a property of one machine's memory bandwidth, not of
TensorForge, and publishing one would turn a measurement into a promise
the project cannot keep. The *only* cross-dtype claim made anywhere is
that equivalent sampler configurations plan the identical index
sequence — which carries no dtype at all.

References, and where a ratio is honest
---------------------------------------

Every case declares exactly one reference type:

- `numpy` — an independently written NumPy expression over the identical
  snapshot, indices, dtype, and output shape, measured through the same
  timing helper. A ratio is published, and what it means is spelled out
  per case in `ratio_meaning`.
- `native_only` — no honest equivalent exists, so **no ratio is
  published at all** and no reference timing is fabricated. Planning,
  permutation construction, materialization, and delivery are all
  `native_only`: NumPy has no batch planner, its shuffle is a different
  algorithm under a different RNG with a different contract, and a case
  that allocates native storage and transfers into it is not the same
  operation as a bare host gather. Dividing by NumPy anywhere in those
  families would compare different things.

What is timed
-------------

One measured sample is exactly one call, timed with
`time.perf_counter_ns()`. Datasets, samplers, loaders, index sets,
iterators, cache state, and restored positions are all built **outside**
the timer, per repetition where the call advances state. Every native
tensor is closed explicitly outside the timer — nothing here relies on
garbage collection. No sample is discarded, no outlier is removed, and no
timer overhead is subtracted. The headline statistic is the **median**;
the spread is the **interquartile range**, stated rather than left to be
inferred, with p25, p75, the minimum, the maximum, and every raw sample
carried in the payload so a reader can recompute anything.

Cold and warm permutation construction are **different cases** and are
never averaged together. A cold case builds a fresh sampler per
repetition, so its cache is empty by construction; the warm case
populates the cache outside the timer and the gate proves the timed call
is a genuine hit — the sampler hands back the *same tuple object*, which
only a cache hit can do. No cache-control API exists, none is added, and
none is needed.

Modes
-----

::

    uv run python benchmarks/benchmark_native_data_pipeline.py
    uv run python benchmarks/benchmark_native_data_pipeline.py --smoke
    uv run python benchmarks/benchmark_native_data_pipeline.py --json
    uv run python benchmarks/benchmark_native_data_pipeline.py --smoke --json
    uv run python benchmarks/benchmark_native_data_pipeline.py --case plan_sequential_exact
    uv run python benchmarks/benchmark_native_data_pipeline.py --workload batch_planning
    uv run python benchmarks/benchmark_native_data_pipeline.py --dtype float32
    uv run python benchmarks/benchmark_native_data_pipeline.py --profile permutation_cold_large

J8 adds **no** capability: no kernel, C ABI export, ctypes declaration,
public API, package export, dtype, device, registry value, checkpoint
field or version, dependency, build option, or CI job. It only measures
what J1-J7 shipped.
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
from tensorforge.experimental import (                      # noqa: E402
    NativeBatchSampler,
    NativeDataLoader,
    NativeTensorDataset,
)

# ---------------------------------------------------------------------------
# Identity. These name the *measurement payload*, not a package export:
# nothing here is added to ``tensorforge.__all__`` or
# ``tensorforge.experimental.__all__``, and there is no benchmark registry
# inside the package.
# ---------------------------------------------------------------------------
BENCHMARK_NAME = "tensorforge.native_data_pipeline"
BENCHMARK_VERSION = "1.0"
# The JSON payload's own contract version. Bumped only when the *shape* of
# the payload changes — never when a measured number does.
SCHEMA_VERSION = 1

# The two publicly supported widths, in the registry's contractual order:
# float64 first, because it is the default an omitted ``dtype`` selects.
DTYPES = ("float64", "float32")
NUMPY_DTYPES = {"float64": np.float64, "float32": np.float32}
BIT_DTYPES = {"float64": np.uint64, "float32": np.uint32}

# ---------------------------------------------------------------------------
# Workload families, in report order. Every one of the four required layers
# is its own family; ``loader_delivery`` is the composition and is reported
# apart from them.
# ---------------------------------------------------------------------------
DATASET_INDEXING = "dataset_indexing"
BATCH_PLANNING = "batch_planning"
PERMUTATION_CONSTRUCTION = "permutation_construction"
HOST_TO_NATIVE = "host_to_native_materialization"
LOADER_DELIVERY = "loader_delivery"
WORKLOADS = (DATASET_INDEXING, BATCH_PLANNING, PERMUTATION_CONSTRUCTION,
             HOST_TO_NATIVE, LOADER_DELIVERY)

# Reference types. ``native_only`` means no honest equivalent exists, so no
# ratio is published for that case anywhere — in the payload or the report.
REFERENCE_NUMPY = "numpy"
NATIVE_ONLY = "native_only"
REFERENCE_TYPES = (REFERENCE_NUMPY, NATIVE_ONLY)

# Correctness gates, one per operation family. Each is exact; none is a
# tolerance.
GATE_HOST_GATHER = "host_gather_bits"
GATE_TARGET_BATCH = "target_batch_exact"
GATE_PLAN = "plan_exact"
GATE_PERMUTATION = "permutation_exact"
GATE_FEATURE_BATCH = "feature_batch_bits"
GATE_DELIVERY = "delivery_transaction"
GATES = (GATE_HOST_GATHER, GATE_TARGET_BATCH, GATE_PLAN, GATE_PERMUTATION,
         GATE_FEATURE_BATCH, GATE_DELIVERY)

# Cache-state labels, for the two permutation cases that deliberately
# measure different cache conditions. ``None`` means the case has no cache
# behaviour to declare.
COLD = "cold"
WARM = "warm"
CACHE_STATES = (COLD, WARM)

# Warm-up and repetition defaults, shared by every case. A case does not
# get to pick its own count: the operations here are all cheap enough that
# one policy is honest for all of them, and the count actually used is
# reported on every record.
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
# Committed reference vectors, transcribed from
# docs/native_data_pipeline_design.md §8.9. These are the *specification*
# of the derivation, not a regression convenience: a case built at one of
# these configurations is gated against the literal, so the permutation and
# planning gates are known-answer checks rather than self-consistency
# checks.
# ---------------------------------------------------------------------------
REFERENCE_PERMUTATIONS = {
    # (length, seed, epoch): order
    (8, 7, 0): (7, 5, 4, 0, 1, 3, 6, 2),
    (8, 7, 7): (1, 4, 7, 0, 3, 5, 6, 2),
}
# §8.9's committed batch plan for length 8, seed 7, epoch 0, batch size 3,
# drop_last False.
REFERENCE_PLAN = ((7, 5, 4), (0, 1, 3), (6, 2))

DISCLAIMER = (
    "Local characterization only, and not a performance contract. These "
    "numbers describe one machine, one build, and one moment; they are not "
    "cross-machine comparable. Correctness is gated before timing, no "
    "duration is ever a pass/fail criterion, no threshold or CI timing job "
    "exists, no result file is written, and no optimization ships with "
    "this measurement. float32 and float64 are measured separately and are "
    "never divided by one another."
)


# ===========================================================================
# Timing and statistics
# ===========================================================================


class Case:
    """One measurable case: a correctness gate, a measured operation, and
    an optional independently written reference operation.

    ``prepare()`` builds whatever one call consumes and runs **outside**
    the timer, once per repetition, so a call that advances state starts
    from identical state every time. ``run(state)`` is the one timed
    operation and nothing else. ``cleanup(state, result)`` releases
    everything the call produced — also outside the timer. ``teardown()``
    releases what the case itself owns.
    """

    __slots__ = ("check", "prepare", "run", "cleanup", "teardown",
                 "reference_prepare", "reference_run", "reference_cleanup")

    def __init__(self, check, prepare, run, cleanup, teardown,
                 reference_prepare=None, reference_run=None,
                 reference_cleanup=None):
        self.check = check
        self.prepare = prepare
        self.run = run
        self.cleanup = cleanup
        self.teardown = teardown
        self.reference_prepare = reference_prepare
        self.reference_run = reference_run
        self.reference_cleanup = reference_cleanup

    @property
    def has_reference(self):
        return self.reference_run is not None


def _no_state():
    """A ``prepare`` for a case whose call consumes nothing per
    repetition. It still runs once per repetition, outside the timer, so
    every case goes through the identical measurement shape."""
    return None


def _discard(state, result):
    """A ``cleanup`` for a call whose result owns nothing releasable.

    The reference is dropped rather than kept: retaining results to
    simplify reporting would hold a whole run's worth of arrays for no
    reason."""
    return None


def measure(prepare, run, cleanup, warmup, repetitions):
    """Warm up, then time exactly ``repetitions`` single calls, in
    nanoseconds.

    One sample is one ``run(state)`` call and nothing else. ``prepare``
    and ``cleanup`` bracket every repetition **outside** the measured
    region, so per-repetition state reconstruction, cache warming,
    position restoration, and native cleanup are never timed. Warm-up
    repetitions run the identical shape and are discarded before
    measuring. **No measured sample is dropped and no timer overhead is
    subtracted.** CPU execution is synchronous, so nothing needs to be
    waited on.
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
# ===========================================================================


def host_features(samples, feature_shape, dtype, seed):
    """Finite, moderate values, **physically at the requested dtype**.

    The array is converted here rather than left for the dataset to
    convert, so the snapshot the dataset copies is byte-identical to the
    one this harness gathers from — which is what lets the indexing gate
    compare the two directly."""
    generator = np.random.default_rng(seed)
    values = generator.uniform(-2.0, 2.0,
                               size=(samples,) + tuple(feature_shape))
    return np.ascontiguousarray(values, dtype=NUMPY_DTYPES[dtype])


def host_targets(samples, classes, seed):
    """Valid non-negative host ``int64`` class labels."""
    generator = np.random.default_rng(seed + 1)
    return np.ascontiguousarray(
        generator.integers(0, classes, size=samples, dtype=np.int64))


def build_dataset(dtype, config, spec):
    """A fresh dataset and the host snapshot it was built from.

    The dtype is **always supplied explicitly** — it is never inferred
    from the array, which is the Phase-I rule this pipeline inherits
    verbatim. Repeated calls build logically equal but entirely
    independent objects.
    """
    samples = config["samples"]
    features = host_features(samples, spec["feature_shape"], dtype,
                             spec["seed"])
    targets = host_targets(samples, spec["classes"], spec["seed"])
    dataset = NativeTensorDataset(features, targets, dtype=dtype)
    return dataset, features, targets


def index_set(pattern, dataset, count, spec):
    """A deterministic index tuple in one of three shapes.

    ``sequential`` is contiguous and increasing; ``shuffled`` comes from a
    real ``NativeBatchSampler`` over the same dataset, so it is the
    pipeline's own order rather than an invented one; ``duplicates``
    repeats each of a shuffled prefix twice, which is a genuinely
    different gather shape (NumPy's advanced indexing cannot answer it
    with a view) and is legal by §10.4.
    """
    if pattern == "sequential":
        return tuple(range(count))
    planner = NativeBatchSampler(dataset, batch_size=count, shuffle=True,
                                 seed=spec["sampler_seed"])
    order = planner.next_batch_indices()
    if pattern == "shuffled":
        return order
    if pattern == "duplicates":
        half = order[: count // 2]
        return tuple(value for entry in half for value in (entry, entry))
    raise ValueError(f"unknown index pattern {pattern!r}")


def bits_of(array, dtype):
    """Raw IEEE-754 bit patterns, with the dtype **asserted rather than
    coerced**.

    A helper that quietly converted could report a match that only existed
    after a conversion this runtime never performs — which is exactly the
    mistake that would make the whole harness dishonest."""
    array = np.asarray(array)
    if array.dtype != NUMPY_DTYPES[dtype]:
        raise AssertionError(
            f"expected a {dtype} array, got {array.dtype}")
    return np.ascontiguousarray(array).reshape(-1).view(BIT_DTYPES[dtype])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# ===========================================================================
# Correctness gates. Every one runs before the timing helper is reached and
# raises AssertionError on failure, which the CLI turns into a nonzero exit
# with clean stdout. All of them are exact: no tolerance appears anywhere.
# ===========================================================================


def gate_host_gather(dataset, snapshot, wanted, dtype, label):
    """The host gather, against an independently indexed snapshot.

    Exactness at four levels: shape, dtype, order (position by position,
    which is what makes duplicate preservation checkable), and raw feature
    bits. Then the gather is tied to the pipeline itself — a real
    ``feature_batch`` over the same indices must carry bit-identical
    values, which is what proves this harness is gathering from the same
    snapshot the dataset owns rather than from a lookalike.
    """
    identity_before = dataset.identity()
    gathered = snapshot[wanted]
    expected_shape = (len(wanted),) + tuple(snapshot.shape[1:])
    require(gathered.shape == expected_shape,
            f"{label}: gathered shape {gathered.shape}, expected "
            f"{expected_shape}")
    require(gathered.dtype == NUMPY_DTYPES[dtype],
            f"{label}: gathered dtype {gathered.dtype}, expected {dtype}")
    require(gathered.flags["C_CONTIGUOUS"],
            f"{label}: the gather is not C-contiguous")
    require(not np.shares_memory(gathered, snapshot),
            f"{label}: the gather aliases the snapshot")
    # Position by position, so a reordered or deduplicated result fails
    # here rather than passing a whole-array comparison by luck.
    for position, index in enumerate(wanted):
        if not np.array_equal(bits_of(gathered[position], dtype),
                              bits_of(snapshot[index], dtype)):
            raise AssertionError(
                f"{label}: row {position} is not the snapshot's row {index}")
    # The pipeline's own materialization of the same request agrees bit
    # for bit, so the snapshot compared above really is the dataset's.
    batch = dataset.feature_batch(wanted)
    try:
        require(np.array_equal(bits_of(batch.to_numpy(), dtype),
                               bits_of(gathered, dtype)),
                f"{label}: the dataset's feature_batch does not match the "
                f"host gather bit for bit")
    finally:
        batch.close()
    require(dataset.identity() == identity_before,
            f"{label}: the dataset changed identity during indexing")
    return {
        "gate": GATE_HOST_GATHER,
        "rows": len(wanted),
        "elements": int(gathered.size),
        "duplicate_indices": len(set(wanted)) != len(wanted),
        "matched_feature_batch_bits": True,
    }


def gate_target_batch(dataset, snapshot, wanted, label):
    """``target_batch``: exact int64 values, and the ownership contract.

    Read-only, C-contiguous, independently owned, fresh at every call, and
    positionally exact — a target batch a caller could edit in place, or
    that aliased the dataset, would break the read-only promise §10.2
    makes."""
    produced = dataset.target_batch(wanted)
    second = dataset.target_batch(wanted)
    expected = np.ascontiguousarray(snapshot[wanted], dtype=np.int64)
    require(produced.dtype == np.int64,
            f"{label}: target dtype {produced.dtype}, expected int64")
    require(produced.shape == (len(wanted),),
            f"{label}: target shape {produced.shape}, expected "
            f"{(len(wanted),)}")
    require(produced.flags["C_CONTIGUOUS"],
            f"{label}: the target batch is not C-contiguous")
    require(produced.flags["WRITEABLE"] is False,
            f"{label}: the target batch is writeable")
    require(not np.shares_memory(produced, snapshot),
            f"{label}: the target batch aliases the dataset's snapshot")
    require(np.array_equal(produced, expected),
            f"{label}: the target batch is not the independently indexed "
            f"host values")
    require(produced is not second and np.array_equal(produced, second),
            f"{label}: repeated calls did not produce equal, independent "
            f"arrays")
    require(not np.shares_memory(produced, second),
            f"{label}: two target batches share memory")
    return {
        "gate": GATE_TARGET_BATCH,
        "rows": len(wanted),
        "elements": int(produced.size),
        "duplicate_indices": len(set(wanted)) != len(wanted),
        "read_only": True,
    }


def _expected_batch_count(samples, batch_size, drop_last):
    """The batch count, in integer arithmetic written independently of the
    sampler's own helper — otherwise the gate would be checking the
    planner against itself."""
    if drop_last:
        return samples // batch_size
    return -(-samples // batch_size)


def gate_plan(sampler, label, expected_plan=None):
    """One epoch's plan, exactly.

    Sequential planning is checked against literal slices of
    ``range(samples)``; shuffled planning against slices of the epoch's
    order, plus — where the configuration matches a §8.9 committed
    vector — against the literal plan itself. Purity is part of the gate:
    the position must not move and a repeated call must return an equal
    plan, because a planner that consumed anything would break the whole
    phase's resume contract.
    """
    samples = sampler.dataset.samples
    batch_size = sampler.batch_size
    before = (sampler.epoch, sampler.cursor)
    count = _expected_batch_count(samples, batch_size, sampler.drop_last)
    plan = sampler.plan()
    require(len(plan) == count,
            f"{label}: plan has {len(plan)} batches, expected {count}")
    require(all(isinstance(group, tuple) for group in plan),
            f"{label}: a plan entry is not a tuple")
    require(all(type(value) is int for group in plan for value in group),
            f"{label}: a plan entry is not an exact int")
    if sampler.shuffle:
        order = sampler.epoch_permutation()
        require(sorted(order) == list(range(samples)),
                f"{label}: the epoch order is not a permutation")
    else:
        order = tuple(range(samples))
        require(sampler.epoch_permutation() == order,
                f"{label}: a sequential sampler did not plan the identity "
                f"order")
    require(plan == tuple(order[k * batch_size:(k + 1) * batch_size]
                          for k in range(count)),
            f"{label}: the plan is not the epoch order sliced by batch size")
    flat = [value for group in plan for value in group]
    require(len(set(flat)) == len(flat),
            f"{label}: an index appears in two batches")
    last = len(plan[-1])
    if sampler.drop_last or samples % batch_size == 0:
        require(last == batch_size,
                f"{label}: the final batch is short at {last}")
    else:
        require(last == samples % batch_size,
                f"{label}: the final batch is {last}, expected "
                f"{samples % batch_size}")
    if expected_plan is not None:
        require(plan == expected_plan,
                f"{label}: the plan does not match the committed reference "
                f"vector")
    require(sampler.plan() == plan,
            f"{label}: two plans of the same position disagree")
    require((sampler.epoch, sampler.cursor) == before,
            f"{label}: planning moved the position")
    return {
        "gate": GATE_PLAN,
        "batches": len(plan),
        "batch_size": batch_size,
        "final_batch": last,
        "short_final_batch": last != batch_size,
        "shuffled": sampler.shuffle,
        "cursor": sampler.cursor,
        "checked_against_reference_vector": expected_plan is not None,
        "position_unchanged": True,
    }


def gate_next_batch_indices(sampler, label):
    """``next_batch_indices()`` is exactly ``plan()[cursor]``, and pure."""
    before = (sampler.epoch, sampler.cursor)
    indices = sampler.next_batch_indices()
    plan = sampler.plan()
    require(indices == plan[sampler.cursor],
            f"{label}: next_batch_indices is not plan()[cursor]")
    require(all(type(value) is int for value in indices),
            f"{label}: an index is not an exact int")
    require(sampler.next_batch_indices() == indices,
            f"{label}: two calls disagree")
    require((sampler.epoch, sampler.cursor) == before,
            f"{label}: the call moved the position")
    return {
        "gate": GATE_PLAN,
        "batches": len(plan),
        "batch_size": sampler.batch_size,
        "final_batch": len(plan[-1]),
        "short_final_batch": len(plan[-1]) != sampler.batch_size,
        "shuffled": sampler.shuffle,
        "cursor": sampler.cursor,
        "checked_against_reference_vector": False,
        "position_unchanged": True,
    }


def gate_permutation(build_sampler, samples, cache_state, label,
                     expected_order=None):
    """One epoch's order, exactly — and proof that the cache state the case
    claims to measure is the state it actually measures.

    **Cold**: two independently constructed samplers produce equal but
    *distinct* tuples, so each of them genuinely computed one. **Warm**: a
    sampler that has already been asked returns the *same tuple object*,
    which only a cache hit can do. Both facts are observable through the
    public surface alone — no cache-control API is used, and none exists.
    """
    first = build_sampler()
    before = (first.epoch, first.cursor)
    order = first.epoch_permutation()
    require(len(order) == samples,
            f"{label}: the order has {len(order)} entries, expected "
            f"{samples}")
    require(sorted(order) == list(range(samples)),
            f"{label}: the order is not a permutation of range({samples})")
    require(all(type(value) is int for value in order),
            f"{label}: an index is not an exact int")
    if expected_order is not None:
        require(order == expected_order,
                f"{label}: the order does not match the committed reference "
                f"vector")
    else:
        # Non-vacuity: a derivation that silently degenerated to the
        # identity would satisfy every structural check above.
        require(order != tuple(range(samples)),
                f"{label}: the shuffled order is the identity, so no "
                f"derivation ran")
    second = build_sampler()
    repeated = second.epoch_permutation()
    require(repeated == order,
            f"{label}: the order is not a pure function of (seed, epoch, "
            f"length)")
    if cache_state == COLD:
        require(repeated is not order,
                f"{label}: a freshly built sampler returned a cached tuple, "
                f"so this case would not be measuring a cold construction")
    else:
        require(first.epoch_permutation() is order,
                f"{label}: a warmed sampler recomputed instead of returning "
                f"its cached order, so this case would not be measuring a "
                f"cache hit")
    require((first.epoch, first.cursor) == before,
            f"{label}: constructing the order moved the position")
    return {
        "gate": GATE_PERMUTATION,
        "length": samples,
        "epoch": first.epoch,
        "cache_state": cache_state,
        "checked_against_reference_vector": expected_order is not None,
        "position_unchanged": True,
    }


def gate_feature_batch(dataset, snapshot, wanted, dtype, label):
    """``feature_batch``: the whole host-to-native materialization.

    dtype, shape, device, ownership, and contiguity by identity; the
    values by raw IEEE-754 bits against an independently gathered host
    array; freshness by producing two batches and proving they are
    distinct owning tensors that survive one another's ``close()``. Every
    tensor the gate creates is closed here, explicitly, before any timing
    exists.
    """
    first = dataset.feature_batch(wanted)
    second = None
    try:
        expected = snapshot[wanted]
        require(first.dtype == dtype,
                f"{label}: batch dtype {first.dtype}, expected {dtype}")
        require(first.shape == expected.shape,
                f"{label}: batch shape {first.shape}, expected "
                f"{expected.shape}")
        require(first.device == "cpu",
                f"{label}: batch device {first.device!r}, expected 'cpu'")
        require(first.owns_core, f"{label}: the batch does not own its core")
        require(first.contiguous, f"{label}: the batch is not contiguous")
        require(first.requires_grad is False,
                f"{label}: a materialized batch requires grad")
        materialized = first.to_numpy()
        require(materialized.dtype == NUMPY_DTYPES[dtype],
                f"{label}: to_numpy widened to {materialized.dtype}")
        require(np.array_equal(bits_of(materialized, dtype),
                               bits_of(expected, dtype)),
                f"{label}: the batch is not bit-identical to the "
                f"independently gathered host values")
        # Egress is a copy: editing what to_numpy() returned must not be
        # visible in a second materialization.
        materialized.reshape(-1)[0] = np.asarray(
            -1.0, dtype=NUMPY_DTYPES[dtype])
        require(np.array_equal(bits_of(first.to_numpy(), dtype),
                               bits_of(expected, dtype)),
                f"{label}: to_numpy() returned a view into native storage")
        second = dataset.feature_batch(wanted)
        require(second is not first,
                f"{label}: a repeated request returned the same object")
        require(np.array_equal(bits_of(second.to_numpy(), dtype),
                               bits_of(expected, dtype)),
                f"{label}: the second batch differs from the first")
        first.close()
        require(second.closed is False,
                f"{label}: closing one batch closed another")
        require(np.array_equal(bits_of(second.to_numpy(), dtype),
                               bits_of(expected, dtype)),
                f"{label}: closing one batch disturbed another")
    finally:
        if not first.closed:
            first.close()
        if second is not None and not second.closed:
            second.close()
    return {
        "gate": GATE_FEATURE_BATCH,
        "rows": len(wanted),
        "elements": int(np.prod(snapshot.shape[1:], dtype=np.int64))
        * len(wanted),
        "duplicate_indices": len(set(wanted)) != len(wanted),
        "device": "cpu",
        "owning_contiguous": True,
        "fresh_storage_per_call": True,
    }


def gate_delivery(loader, snapshot, targets_snapshot, canonical, dtype,
                  label):
    """One whole ``next(iterator)``: the §9.4 transaction, end to end.

    The candidate indices are recorded **before** the delivery, exactly as
    a caller must; the delivered feature tensor and target array are then
    gated against independently gathered host values; and the committed
    position must have advanced by exactly one batch — the invariant the
    whole phase rests on. Everything is closed and the canonical position
    restored before the gate returns, so timing starts from the same state
    the gate started from.
    """
    sampler = loader.sampler
    before = (sampler.epoch, sampler.cursor)
    expected_indices = sampler.next_batch_indices()
    iterator = iter(loader)
    features = None
    try:
        features, targets = next(iterator)
        expected = snapshot[list(expected_indices)]
        require(features.dtype == dtype,
                f"{label}: delivered dtype {features.dtype}, expected "
                f"{dtype}")
        require(features.shape == expected.shape,
                f"{label}: delivered shape {features.shape}, expected "
                f"{expected.shape}")
        require(features.device == "cpu",
                f"{label}: delivered device {features.device!r}")
        require(features.owns_core and features.contiguous,
                f"{label}: the delivered batch is not owning and contiguous")
        require(np.array_equal(bits_of(features.to_numpy(), dtype),
                               bits_of(expected, dtype)),
                f"{label}: the delivered batch is not the planned rows")
        require(targets.dtype == np.int64 and not targets.flags["WRITEABLE"],
                f"{label}: the delivered targets are not read-only int64")
        require(np.array_equal(
            targets, np.ascontiguousarray(
                targets_snapshot[list(expected_indices)], dtype=np.int64)),
            f"{label}: the delivered targets are not the planned labels")
        expected_after = (before[0], before[1] + 1)
        if expected_after[1] == sampler.batches_per_epoch:
            expected_after = (before[0] + 1, 0)
        require((sampler.epoch, sampler.cursor) == expected_after,
                f"{label}: a delivered batch moved the position to "
                f"{(sampler.epoch, sampler.cursor)}, expected "
                f"{expected_after}")
    finally:
        if features is not None:
            features.close()
        iterator.close()
        loader.load_state_dict(canonical)
    return {
        "gate": GATE_DELIVERY,
        "rows": len(expected_indices),
        "batch_size": sampler.batch_size,
        "position_advanced_by_one_batch": True,
        "restored_to_canonical_position": True,
    }


# ===========================================================================
# Case builders. Each returns a Case whose gate has already been written
# against an independent oracle; none of them times anything.
# ===========================================================================


def build_host_feature_gather(dtype, config, spec):
    """The §10.4 M1 host gather, in isolation.

    This is the *host* half of materialization: no native storage is
    allocated, nothing is transferred, and no dataset method is timed. Its
    reference is a second, independently written NumPy spelling of the
    identical gather, so the ratio is a NumPy-internal observation about
    two ways of writing one operation — **not** a TensorForge-versus-NumPy
    comparison, and the payload says so.
    """
    dataset, features, _ = build_dataset(dtype, config, spec)
    wanted = list(index_set(spec["index_pattern"], dataset, config["batch"],
                            spec))
    # Byte-identical to the dataset's own snapshot, and proved so by the
    # gate through a real feature_batch.
    snapshot = np.array(features, dtype=NUMPY_DTYPES[dtype], order="C",
                        copy=True)
    label = spec["label"]

    def run(state):
        return snapshot[wanted]

    def reference_run(state):
        return np.take(snapshot, wanted, axis=0)

    def check():
        metrics = gate_host_gather(dataset, snapshot, wanted, dtype, label)
        require(np.array_equal(bits_of(run(None), dtype),
                               bits_of(reference_run(None), dtype)),
                f"{label}: the reference spelling of the gather disagrees "
                f"with the measured one")
        metrics["reference_agrees"] = True
        return metrics

    return Case(check, _no_state, run, _discard, dataset.close,
                reference_prepare=_no_state, reference_run=reference_run,
                reference_cleanup=_discard)


def build_target_batch(dtype, config, spec):
    """``dataset.target_batch(indices)`` — the one public host-only
    indexing method the pipeline has.

    Its reference is the same host gather and copy written independently,
    without the dataset's index validation and lifecycle guard, so the
    ratio is exactly the cost of that validation and dispatch.
    """
    dataset, _, targets = build_dataset(dtype, config, spec)
    wanted = list(index_set(spec["index_pattern"], dataset, config["batch"],
                            spec))
    snapshot = np.array(targets, dtype=np.int64, order="C", copy=True)
    label = spec["label"]

    def run(state):
        return dataset.target_batch(wanted)

    def reference_run(state):
        gathered = np.ascontiguousarray(snapshot[wanted], dtype=np.int64)
        gathered.setflags(write=False)
        return gathered

    def check():
        metrics = gate_target_batch(dataset, snapshot, wanted, label)
        produced = run(None)
        reference = reference_run(None)
        require(np.array_equal(produced, reference),
                f"{label}: the reference gather disagrees with the dataset's "
                f"target batch")
        require(reference.flags["WRITEABLE"] is False,
                f"{label}: the reference is not read-only, so it is not the "
                f"same operation")
        metrics["reference_agrees"] = True
        return metrics

    return Case(check, _no_state, run, _discard, dataset.close,
                reference_prepare=_no_state, reference_run=reference_run,
                reference_cleanup=_discard)


def _sampler_at(dataset, config, spec):
    """A sampler at the case's configuration, positioned through the
    public state loader when the case wants a non-initial position.

    Constructing and positioning are setup: they happen here, outside
    every timer.
    """
    sampler = NativeBatchSampler(dataset, batch_size=config["batch_size"],
                                 shuffle=spec["shuffle"],
                                 seed=spec["sampler_seed"],
                                 drop_last=spec["drop_last"])
    epoch = spec.get("epoch", 0)
    cursor = config.get("cursor", 0)
    if epoch or cursor:
        state = sampler.state_dict()
        state["epoch"] = epoch
        state["cursor"] = cursor
        sampler.load_state_dict(state)
    return sampler


def build_plan(dtype, config, spec):
    """``sampler.plan()`` — a whole epoch's batch grouping.

    ``native_only``: NumPy has no batch planner, and inventing one to
    divide by would be measuring code written for the benchmark rather
    than code the project ships.
    """
    dataset, _, _ = build_dataset(dtype, config, spec)
    sampler = _sampler_at(dataset, config, spec)
    expected_plan = spec.get("reference_plan")
    label = spec["label"]

    def run(state):
        return sampler.plan()

    def check():
        return gate_plan(sampler, label, expected_plan)

    return Case(check, _no_state, run, _discard, dataset.close)


def build_next_batch_indices(dtype, config, spec):
    """``sampler.next_batch_indices()`` — the one group a loader is about
    to use, computed without building the rest of the plan.

    ``native_only``, for the same reason planning is.
    """
    dataset, _, _ = build_dataset(dtype, config, spec)
    sampler = _sampler_at(dataset, config, spec)
    label = spec["label"]

    def run(state):
        return sampler.next_batch_indices()

    def check():
        return gate_next_batch_indices(sampler, label)

    return Case(check, _no_state, run, _discard, dataset.close)


def build_permutation(dtype, config, spec):
    """``sampler.epoch_permutation()`` — the deterministic shuffled order.

    ``native_only``, and this one is not a close call: NumPy's shuffle is
    a different algorithm driven by a different generator with a different
    contract, so a ratio against it would divide two things that are not
    the same operation and cannot produce the same answer.

    A **cold** case builds a fresh sampler per repetition — outside the
    timer — so the cache is empty by construction and the timed call
    genuinely derives the order. A **warm** case populates the cache
    outside the timer and times a genuine hit. The two are separate cases
    and are never averaged together.
    """
    dataset, _, _ = build_dataset(dtype, config, spec)
    cache_state = spec["cache_state"]
    label = spec["label"]

    def fresh():
        return _sampler_at(dataset, config, spec)

    if cache_state == COLD:
        def prepare():
            # A brand-new sampler: no cached order exists, so the timed
            # call below must construct one.
            return fresh()
    else:
        warm = fresh()

        def prepare():
            # Populate the cache here, outside the timer, so the measured
            # call is a hit rather than a construction.
            warm.epoch_permutation()
            return warm

    def run(sampler):
        return sampler.epoch_permutation()

    def check():
        return gate_permutation(fresh, config["samples"], cache_state, label,
                                spec.get("reference_permutation"))

    return Case(check, prepare, run, _discard, dataset.close)


def build_feature_batch(dtype, config, spec):
    """``dataset.feature_batch(indices)`` — the whole host-to-native path:
    one host gather, one fresh owning native allocation, and one transfer.

    ``native_only``. A NumPy fancy index performs the gather and nothing
    else, so dividing this by one would credit or blame TensorForge for an
    allocation and a transfer the reference never made. The host gather is
    measured honestly, against NumPy, in the ``dataset_indexing`` family
    instead.
    """
    dataset, features, _ = build_dataset(dtype, config, spec)
    wanted = list(index_set(spec["index_pattern"], dataset, config["batch"],
                            spec))
    snapshot = np.array(features, dtype=NUMPY_DTYPES[dtype], order="C",
                        copy=True)
    label = spec["label"]

    def run(state):
        return dataset.feature_batch(wanted)

    def cleanup(state, result):
        # Explicit, and outside the timer. Nothing here waits for garbage
        # collection, and no batch is retained to simplify reporting.
        if result is not None:
            result.close()

    def check():
        return gate_feature_batch(dataset, snapshot, wanted, dtype, label)

    return Case(check, _no_state, run, cleanup, dataset.close)


def build_loader_delivery(dtype, config, spec):
    """One successful ``next(iterator)`` — the §9.4 five-phase transaction.

    Deliberately a **composition** and reported as one: it is planning
    plus a host gather plus a native allocation plus a transfer plus a
    target copy plus the transaction bookkeeping, and no part of it is
    isolated here. It exists because §22.3 makes "is the gather plus copy
    the dominant cost of a realistic step?" the only question that could
    reopen the export count, and that question needs both the composed
    number and the isolated layers above it.

    ``native_only``, and it must be: there is no reference implementation
    of a transactional batch handoff to divide by.

    The call advances the committed position, so every repetition restores
    the canonical position through the public loader state — outside the
    timer — and closes both the delivered tensor and the iterator, also
    outside the timer.
    """
    dataset, features, targets = build_dataset(dtype, config, spec)
    sampler = _sampler_at(dataset, config, spec)
    loader = NativeDataLoader(sampler)
    snapshot = np.array(features, dtype=NUMPY_DTYPES[dtype], order="C",
                        copy=True)
    target_snapshot = np.array(targets, dtype=np.int64, order="C", copy=True)
    canonical = loader.state_dict()
    label = spec["label"]

    def prepare():
        # Restore the exact starting position, then warm the epoch's order
        # and open a fresh one-epoch iterator. All three are setup and all
        # three are outside the timer.
        #
        # The warming matters: a state load invalidates the permutation
        # cache, so without it the first delivery of every repetition would
        # rebuild the whole epoch order inside the timer and this case
        # would silently be measuring permutation construction — which
        # already has four cases of its own — rather than the handoff.
        loader.load_state_dict(canonical)
        sampler.epoch_permutation()
        return iter(loader)

    def run(iterator):
        return next(iterator)

    def cleanup(iterator, result):
        if result is not None:
            result[0].close()
        iterator.close()

    def check():
        return gate_delivery(loader, snapshot, target_snapshot, canonical,
                             dtype, label)

    def teardown():
        loader.close()
        dataset.close()

    return Case(check, prepare, run, cleanup, teardown)


# ===========================================================================
# The case registry
#
# One exact, ordered inventory. Every case declares enough metadata to be
# audited without reading its builder: what it measures, what it compares
# against, what is set up outside the timer, what the timer contains, what
# is cleaned up, whether it is native-only, and which cache state it
# deliberately measures.
# ===========================================================================

_GEOMETRY_VECTOR = (24,)          # a 1-D per-sample feature vector
_GEOMETRY_IMAGE = (1, 8, 8)       # a rank-3 per-sample block, NCHW-shaped

_HOST_ONLY_SETUP = ("the dataset, its host snapshot, and the index tuple "
                    "are built once, outside the timer")
_HOST_ONLY_CLEANUP = ("the gather result owns nothing releasable and is "
                      "dropped outside the timer; the dataset is closed at "
                      "teardown")
_PLANNER_SETUP = ("the dataset, the sampler, and any restored position are "
                  "built outside the timer")
_PLANNER_CLEANUP = ("the plan is a tuple of tuples of ints and owns nothing "
                    "releasable; the dataset is closed at teardown")

CASES = {
    # -- dataset indexing --------------------------------------------------
    "host_feature_gather_sequential": {
        "workload": DATASET_INDEXING,
        "label": "host_feature_gather_sequential",
        "operation": ("the §10.4 M1 host gather: snapshot[indices] over a "
                      "contiguous increasing index set"),
        "build": build_host_feature_gather,
        "gate": GATE_HOST_GATHER,
        "reference_type": REFERENCE_NUMPY,
        "native_only": False,
        "reference_detail": ("numpy.take(snapshot, indices, axis=0) — a "
                             "second, independently written spelling of the "
                             "identical gather over the identical snapshot, "
                             "indices, dtype, and output shape"),
        "ratio_meaning": ("measured median / reference median. Both sides "
                          "are NumPy expressions over the same host memory, "
                          "so this is an observation about two spellings of "
                          "one gather and is not a TensorForge-versus-NumPy "
                          "comparison."),
        "correctness_reference": ("an independently indexed NumPy snapshot, "
                                  "position by position, in raw IEEE-754 "
                                  "bits, plus the dataset's own "
                                  "feature_batch over the same indices"),
        "seed": 20260401,
        "sampler_seed": 11,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": "sequential",
        "shuffle": False,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 8},
            "full": {"samples": 4096, "batch": 256},
            "profile": {"samples": 16384, "batch": 1024},
        },
        "setup": _HOST_ONLY_SETUP,
        "timed": "exactly one host gather",
        "cleanup": _HOST_ONLY_CLEANUP,
        "notes": ("The host half of materialization, isolated. No native "
                  "storage is allocated and nothing is transferred, which "
                  "is what makes the NumPy reference honest here and "
                  "dishonest in the materialization family."),
    },
    "host_feature_gather_shuffled": {
        "workload": DATASET_INDEXING,
        "label": "host_feature_gather_shuffled",
        "operation": ("the §10.4 M1 host gather over a deterministic "
                      "shuffled index set taken from a real sampler"),
        "build": build_host_feature_gather,
        "gate": GATE_HOST_GATHER,
        "reference_type": REFERENCE_NUMPY,
        "native_only": False,
        "reference_detail": ("numpy.take(snapshot, indices, axis=0) over the "
                             "identical snapshot and the identical shuffled "
                             "indices"),
        "ratio_meaning": ("measured median / reference median, two spellings "
                          "of one NumPy gather over the same host memory"),
        "correctness_reference": ("an independently indexed NumPy snapshot, "
                                  "position by position, in raw IEEE-754 "
                                  "bits"),
        "seed": 20260402,
        "sampler_seed": 12,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": "shuffled",
        "shuffle": True,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 8},
            "full": {"samples": 4096, "batch": 256},
            "profile": {"samples": 16384, "batch": 1024},
        },
        "setup": _HOST_ONLY_SETUP,
        "timed": "exactly one host gather",
        "cleanup": _HOST_ONLY_CLEANUP,
        "notes": ("The scattered-read shape a shuffled epoch actually "
                  "produces, which is the one a training run pays for."),
    },
    "host_feature_gather_duplicates": {
        "workload": DATASET_INDEXING,
        "label": "host_feature_gather_duplicates",
        "operation": ("the §10.4 M1 host gather over an index set in which "
                      "every index appears twice"),
        "build": build_host_feature_gather,
        "gate": GATE_HOST_GATHER,
        "reference_type": REFERENCE_NUMPY,
        "native_only": False,
        "reference_detail": ("numpy.take(snapshot, indices, axis=0) over the "
                             "identical duplicated index set"),
        "ratio_meaning": ("measured median / reference median, two spellings "
                          "of one NumPy gather over the same host memory"),
        "correctness_reference": ("an independently indexed NumPy snapshot, "
                                  "position by position, so a deduplicated "
                                  "or reordered result cannot pass"),
        "seed": 20260403,
        "sampler_seed": 13,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": "duplicates",
        "shuffle": True,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 8},
            "full": {"samples": 4096, "batch": 256},
            "profile": {"samples": 16384, "batch": 1024},
        },
        "setup": _HOST_ONLY_SETUP,
        "timed": "exactly one host gather",
        "cleanup": _HOST_ONLY_CLEANUP,
        "notes": ("A distinct gather shape: repeated rows cannot be answered "
                  "with a view. §10.4 permits duplicates and the sampler "
                  "never produces one, so this characterizes the contract "
                  "rather than a path a training run takes."),
    },
    "dataset_target_batch_sequential": {
        "workload": DATASET_INDEXING,
        "label": "dataset_target_batch_sequential",
        "operation": ("NativeTensorDataset.target_batch(indices) over a "
                      "contiguous increasing index set"),
        "build": build_target_batch,
        "gate": GATE_TARGET_BATCH,
        "reference_type": REFERENCE_NUMPY,
        "native_only": False,
        "reference_detail": ("numpy.ascontiguousarray(snapshot[indices], "
                             "dtype=int64) with setflags(write=False) — the "
                             "same gather, copy, and read-only publication, "
                             "written without the dataset's index validation "
                             "and lifecycle guard"),
        "ratio_meaning": ("measured median / reference median, so the ratio "
                          "is the cost of the public method's index "
                          "validation and dispatch over the identical host "
                          "work"),
        "correctness_reference": ("exact int64 equality with an "
                                  "independently indexed host snapshot, plus "
                                  "the contiguity, ownership, and read-only "
                                  "contract"),
        "seed": 20260404,
        "sampler_seed": 14,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": "sequential",
        "shuffle": False,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 8},
            "full": {"samples": 4096, "batch": 256},
            "profile": {"samples": 16384, "batch": 1024},
        },
        "setup": _HOST_ONLY_SETUP,
        "timed": "exactly one target_batch call",
        "cleanup": _HOST_ONLY_CLEANUP,
        "notes": ("Targets are host int64 metadata at every feature dtype — "
                  "no native integer tensor exists — so the two dtype rows "
                  "of this case differ only in the dataset they were taken "
                  "from, and neither is compared to the other."),
    },
    "dataset_target_batch_shuffled": {
        "workload": DATASET_INDEXING,
        "label": "dataset_target_batch_shuffled",
        "operation": ("NativeTensorDataset.target_batch(indices) over a "
                      "deterministic shuffled index set"),
        "build": build_target_batch,
        "gate": GATE_TARGET_BATCH,
        "reference_type": REFERENCE_NUMPY,
        "native_only": False,
        "reference_detail": ("the same independently written gather, copy, "
                             "and read-only publication over the identical "
                             "shuffled indices"),
        "ratio_meaning": ("measured median / reference median, the public "
                          "method's validation and dispatch over identical "
                          "host work"),
        "correctness_reference": ("exact int64 equality with an "
                                  "independently indexed host snapshot, plus "
                                  "the ownership and read-only contract"),
        "seed": 20260405,
        "sampler_seed": 15,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": "shuffled",
        "shuffle": True,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 8},
            "full": {"samples": 4096, "batch": 256},
            "profile": {"samples": 16384, "batch": 1024},
        },
        "setup": _HOST_ONLY_SETUP,
        "timed": "exactly one target_batch call",
        "cleanup": _HOST_ONLY_CLEANUP,
        "notes": ("The label half of the shuffled batch a training step "
                  "consumes beside its features."),
    },
    # -- batch planning ----------------------------------------------------
    "plan_sequential_exact": {
        "workload": BATCH_PLANNING,
        "label": "plan_sequential_exact",
        "operation": ("NativeBatchSampler.plan() with shuffle=False and a "
                      "batch size that divides the sample count exactly"),
        "build": build_plan,
        "gate": GATE_PLAN,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — NumPy has no batch planner, and writing "
                             "one for the benchmark would time code the "
                             "project does not ship"),
        "ratio_meaning": None,
        "correctness_reference": ("literal slices of range(samples), plus an "
                                  "independently computed batch count"),
        "seed": 20260406,
        "sampler_seed": 21,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": False,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 128, "batch_size": 32},
            "full": {"samples": 4096, "batch_size": 64},
            "profile": {"samples": 16384, "batch_size": 64},
        },
        "setup": _PLANNER_SETUP,
        "timed": "exactly one plan() call",
        "cleanup": _PLANNER_CLEANUP,
        "notes": ("Sequential planning is a different branch from shuffled "
                  "planning — it consumes no derivation at all — so the two "
                  "are separate cases rather than one with a flag. It is "
                  "also the branch with no cache: the identity order is "
                  "rebuilt on every call, where a shuffled active-epoch "
                  "order is not. That is a property of the two branches and "
                  "is reported as one; nothing here proposes changing it."),
    },
    "plan_sequential_short_final": {
        "workload": BATCH_PLANNING,
        "label": "plan_sequential_short_final",
        "operation": ("NativeBatchSampler.plan() with shuffle=False, "
                      "drop_last=False, and a batch size that leaves a short "
                      "final batch"),
        "build": build_plan,
        "gate": GATE_PLAN,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": "none — there is no batch planner to divide by",
        "ratio_meaning": None,
        "correctness_reference": ("literal slices of range(samples), with the "
                                  "final batch's length checked against "
                                  "samples % batch_size"),
        "seed": 20260407,
        "sampler_seed": 22,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": False,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 130, "batch_size": 32},
            "full": {"samples": 4100, "batch_size": 64},
            "profile": {"samples": 16388, "batch_size": 64},
        },
        "setup": _PLANNER_SETUP,
        "timed": "exactly one plan() call",
        "cleanup": _PLANNER_CLEANUP,
        "notes": ("The truncating final slice is the specified behaviour, "
                  "not an edge case, so it is measured rather than avoided."),
    },
    "plan_shuffled_reference": {
        "workload": BATCH_PLANNING,
        "label": "plan_shuffled_reference",
        "operation": ("NativeBatchSampler.plan() at the §8.9 committed "
                      "configuration: 8 samples, seed 7, epoch 0, batch "
                      "size 3"),
        "build": build_plan,
        "gate": GATE_PLAN,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": "none — there is no batch planner to divide by",
        "ratio_meaning": None,
        "correctness_reference": ("the committed §8.9 plan "
                                  "((7, 5, 4), (0, 1, 3), (6, 2)), as a "
                                  "literal known answer"),
        "seed": 20260408,
        "sampler_seed": 7,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "epoch": 0,
        "reference_plan": REFERENCE_PLAN,
        "cache_state": None,
        "fixed_configuration": True,
        "configurations": {
            "smoke": {"samples": 8, "batch_size": 3},
            "full": {"samples": 8, "batch_size": 3},
            "profile": {"samples": 8, "batch_size": 3},
        },
        "setup": _PLANNER_SETUP,
        "timed": "exactly one plan() call",
        "cleanup": _PLANNER_CLEANUP,
        "notes": ("Deliberately tiny and identical in every configuration: "
                  "its whole value is that the timed operation's own result "
                  "is checkable against a committed literal, which a larger "
                  "shape could not be. It also shows the fixed per-call "
                  "floor a plan cannot go below."),
    },
    "plan_shuffled_large": {
        "workload": BATCH_PLANNING,
        "label": "plan_shuffled_large",
        "operation": ("NativeBatchSampler.plan() with shuffle=True over a "
                      "realistic sample count"),
        "build": build_plan,
        "gate": GATE_PLAN,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": "none — there is no batch planner to divide by",
        "ratio_meaning": None,
        "correctness_reference": ("slices of the epoch's own order, which is "
                                  "independently checked to be a permutation "
                                  "of range(samples)"),
        "seed": 20260409,
        "sampler_seed": 23,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 128, "batch_size": 32},
            "full": {"samples": 4096, "batch_size": 64},
            "profile": {"samples": 16384, "batch_size": 64},
        },
        "setup": _PLANNER_SETUP,
        "timed": "exactly one plan() call",
        "cleanup": _PLANNER_CLEANUP,
        "notes": ("The sampler is constructed fresh in setup, so the first "
                  "measured call may construct the epoch order and later "
                  "ones read the cache. Cold and warm construction are "
                  "isolated in the permutation family instead; what this "
                  "case measures is the grouping a caller sees."),
    },
    "next_batch_indices_fresh": {
        "workload": BATCH_PLANNING,
        "label": "next_batch_indices_fresh",
        "operation": ("NativeBatchSampler.next_batch_indices() at a fresh "
                      "position (epoch 0, cursor 0)"),
        "build": build_next_batch_indices,
        "gate": GATE_PLAN,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": "none — there is no batch planner to divide by",
        "ratio_meaning": None,
        "correctness_reference": "plan()[cursor], exactly",
        "seed": 20260410,
        "sampler_seed": 24,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 128, "batch_size": 32},
            "full": {"samples": 4096, "batch_size": 64},
            "profile": {"samples": 16384, "batch_size": 64},
        },
        "setup": _PLANNER_SETUP,
        "timed": "exactly one next_batch_indices() call",
        "cleanup": _PLANNER_CLEANUP,
        "notes": ("The call a loader makes once per batch, and the one J6's "
                  "example records before every delivery. It computes one "
                  "slice rather than the whole plan."),
    },
    "next_batch_indices_mid_epoch": {
        "workload": BATCH_PLANNING,
        "label": "next_batch_indices_mid_epoch",
        "operation": ("NativeBatchSampler.next_batch_indices() at a genuine "
                      "mid-epoch cursor restored through the public state "
                      "loader"),
        "build": build_next_batch_indices,
        "gate": GATE_PLAN,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": "none — there is no batch planner to divide by",
        "ratio_meaning": None,
        "correctness_reference": "plan()[cursor] at the restored cursor",
        "seed": 20260411,
        "sampler_seed": 25,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 128, "batch_size": 32, "cursor": 1},
            "full": {"samples": 4096, "batch_size": 64, "cursor": 5},
            "profile": {"samples": 16384, "batch_size": 64, "cursor": 5},
        },
        "setup": ("the dataset and sampler are built and the mid-epoch "
                  "position restored through load_state_dict, all outside "
                  "the timer"),
        "timed": "exactly one next_batch_indices() call",
        "cleanup": _PLANNER_CLEANUP,
        "notes": ("A resumed run spends its first epoch here, so the "
                  "mid-epoch position is measured rather than assumed to "
                  "cost what a fresh one does."),
    },
    # -- permutation construction ------------------------------------------
    "permutation_cold_reference": {
        "workload": PERMUTATION_CONSTRUCTION,
        "label": "permutation_cold_reference",
        "operation": ("epoch_permutation() on a freshly built sampler at the "
                      "§8.9 committed configuration: 8 samples, seed 7, "
                      "epoch 0"),
        "build": build_permutation,
        "gate": GATE_PERMUTATION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — NumPy's shuffle is a different "
                             "algorithm under a different generator with a "
                             "different contract, so a ratio would divide "
                             "two operations that cannot produce the same "
                             "answer"),
        "ratio_meaning": None,
        "correctness_reference": ("the committed §8.9 vector "
                                  "(7, 5, 4, 0, 1, 3, 6, 2), as a literal "
                                  "known answer"),
        "seed": 20260412,
        "sampler_seed": 7,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "epoch": 0,
        "reference_permutation": REFERENCE_PERMUTATIONS[(8, 7, 0)],
        "cache_state": COLD,
        "fixed_configuration": True,
        "configurations": {
            "smoke": {"samples": 8, "batch_size": 3},
            "full": {"samples": 8, "batch_size": 3},
            "profile": {"samples": 8, "batch_size": 3},
        },
        "setup": ("a brand-new sampler is constructed per repetition, "
                  "outside the timer, so its permutation cache is empty by "
                  "construction"),
        "timed": "exactly one epoch_permutation() call, on a cold cache",
        "cleanup": ("the order is a tuple of ints and owns nothing "
                    "releasable; the dataset is closed at teardown"),
        "notes": ("Small and fixed on purpose: the timed call's own result "
                  "is checkable against a committed literal. Seven draws is "
                  "also the shortest honest cold construction, so this row "
                  "is close to the fixed per-call floor."),
    },
    "permutation_cold_later_epoch": {
        "workload": PERMUTATION_CONSTRUCTION,
        "label": "permutation_cold_later_epoch",
        "operation": ("epoch_permutation() on a freshly built sampler "
                      "restored to epoch 7, at the §8.9 committed "
                      "configuration"),
        "build": build_permutation,
        "gate": GATE_PERMUTATION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — NumPy's shuffle is a different "
                             "algorithm under a different generator"),
        "ratio_meaning": None,
        "correctness_reference": ("the committed §8.9 vector "
                                  "(1, 4, 7, 0, 3, 5, 6, 2) for epoch 7, as "
                                  "a literal known answer"),
        "seed": 20260413,
        "sampler_seed": 7,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "epoch": 7,
        "reference_permutation": REFERENCE_PERMUTATIONS[(8, 7, 7)],
        "cache_state": COLD,
        "fixed_configuration": True,
        "configurations": {
            "smoke": {"samples": 8, "batch_size": 3},
            "full": {"samples": 8, "batch_size": 3},
            "profile": {"samples": 8, "batch_size": 3},
        },
        "setup": ("a brand-new sampler is constructed and restored to "
                  "epoch 7 per repetition, outside the timer"),
        "timed": "exactly one epoch_permutation() call, on a cold cache",
        "cleanup": ("the order is a tuple of ints; the dataset is closed at "
                    "teardown"),
        "notes": ("A later epoch costs exactly what epoch 0 costs — the "
                  "epoch enters through one key derivation, not through a "
                  "replay of the epochs before it — and this case is where "
                  "that is measured rather than asserted."),
    },
    "permutation_cold_large": {
        "workload": PERMUTATION_CONSTRUCTION,
        "label": "permutation_cold_large",
        "operation": ("epoch_permutation() on a freshly built sampler over a "
                      "realistic sample count, with an empty cache"),
        "build": build_permutation,
        "gate": GATE_PERMUTATION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — NumPy's shuffle is a different "
                             "algorithm under a different generator"),
        "ratio_meaning": None,
        "correctness_reference": ("a permutation of range(samples), equal "
                                  "across independently constructed "
                                  "samplers, and provably not the identity"),
        "seed": 20260414,
        "sampler_seed": 26,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "cache_state": COLD,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 128, "batch_size": 32},
            "full": {"samples": 4096, "batch_size": 64},
            "profile": {"samples": 16384, "batch_size": 64},
        },
        "setup": ("a brand-new sampler is constructed per repetition, "
                  "outside the timer, so the cache is empty by construction"),
        "timed": "exactly one epoch_permutation() call, on a cold cache",
        "cleanup": ("the order is a tuple of ints; the dataset is closed at "
                    "teardown"),
        "notes": ("The once-per-epoch cost of the derivation, at a size a "
                  "training run would actually use. It is pure Python "
                  "integer arithmetic by design — that is what makes it "
                  "bit-identical on every platform — so this is the case "
                  "that says what determinism costs."),
    },
    "permutation_cache_hit": {
        "workload": PERMUTATION_CONSTRUCTION,
        "label": "permutation_cache_hit",
        "operation": ("epoch_permutation() on a sampler whose active-epoch "
                      "order is already cached"),
        "build": build_permutation,
        "gate": GATE_PERMUTATION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — a cache lookup on a private tuple has "
                             "no NumPy equivalent to divide by"),
        "ratio_meaning": None,
        "correctness_reference": ("the same order the cold case produces, "
                                  "returned as the identical tuple object — "
                                  "which only a cache hit can do"),
        "seed": 20260415,
        "sampler_seed": 27,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "cache_state": WARM,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 128, "batch_size": 32},
            "full": {"samples": 4096, "batch_size": 64},
            "profile": {"samples": 16384, "batch_size": 64},
        },
        "setup": ("the cache is populated outside the timer, once per "
                  "repetition, so the measured call is a genuine hit"),
        "timed": "exactly one epoch_permutation() call, on a warm cache",
        "cleanup": ("the order is a tuple of ints; the dataset is closed at "
                    "teardown"),
        "notes": ("Reported as its own case and **never averaged with the "
                  "cold one**: the two answer different questions, and one "
                  "number covering both would describe neither. The cache is "
                  "a private tuple of ints, holds no native storage, and is "
                  "reachable through no public control."),
    },
    # -- host-to-native materialization ------------------------------------
    "feature_batch_small": {
        "workload": HOST_TO_NATIVE,
        "label": "feature_batch_small",
        "operation": ("dataset.feature_batch(indices) for a small batch of "
                      "contiguous increasing indices"),
        "build": build_feature_batch,
        "gate": GATE_FEATURE_BATCH,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — a NumPy fancy index performs the gather "
                             "and nothing else, so dividing by it would "
                             "charge TensorForge for an allocation and a "
                             "transfer the reference never made. The gather "
                             "alone is measured against NumPy in the "
                             "dataset_indexing family instead"),
        "ratio_meaning": None,
        "correctness_reference": ("raw IEEE-754 bit equality with an "
                                  "independently gathered host array, plus "
                                  "the dtype, shape, device, ownership, "
                                  "contiguity, and freshness contract"),
        "seed": 20260416,
        "sampler_seed": 31,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": "sequential",
        "shuffle": False,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 8},
            "full": {"samples": 4096, "batch": 32},
            "profile": {"samples": 16384, "batch": 64},
        },
        "setup": ("the dataset, its host snapshot, and the index list are "
                  "built once, outside the timer"),
        "timed": ("exactly one feature_batch call: the host gather, the "
                  "fresh owning native allocation, and the transfer"),
        "cleanup": ("the returned NativeTensor is closed explicitly after "
                    "every repetition, outside the timer; the dataset is "
                    "closed at teardown"),
        "notes": ("Small batches are where the fixed per-call Python and "
                  "ctypes cost is visible. That is an architectural floor "
                  "rather than a defect, and this case exists to show it."),
    },
    "feature_batch_large": {
        "workload": HOST_TO_NATIVE,
        "label": "feature_batch_large",
        "operation": ("dataset.feature_batch(indices) for a large batch of "
                      "contiguous increasing indices"),
        "build": build_feature_batch,
        "gate": GATE_FEATURE_BATCH,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the native case allocates and transfers "
                             "and a NumPy gather does not"),
        "ratio_meaning": None,
        "correctness_reference": ("raw IEEE-754 bit equality with an "
                                  "independently gathered host array, plus "
                                  "the ownership and freshness contract"),
        "seed": 20260417,
        "sampler_seed": 32,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": "sequential",
        "shuffle": False,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 16},
            "full": {"samples": 4096, "batch": 512},
            "profile": {"samples": 16384, "batch": 2048},
        },
        "setup": ("the dataset, its host snapshot, and the index list are "
                  "built once, outside the timer"),
        "timed": "exactly one feature_batch call",
        "cleanup": ("the returned NativeTensor is closed explicitly after "
                    "every repetition, outside the timer"),
        "notes": ("Where the transfer rather than the fixed cost dominates. "
                  "Each dtype is characterized on its own; the two are never "
                  "divided by one another, whatever the element widths "
                  "suggest."),
    },
    "feature_batch_shuffled": {
        "workload": HOST_TO_NATIVE,
        "label": "feature_batch_shuffled",
        "operation": ("dataset.feature_batch(indices) for a large batch of "
                      "deterministic shuffled indices"),
        "build": build_feature_batch,
        "gate": GATE_FEATURE_BATCH,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the native case allocates and transfers "
                             "and a NumPy gather does not"),
        "ratio_meaning": None,
        "correctness_reference": ("raw IEEE-754 bit equality with an "
                                  "independently gathered host array, in the "
                                  "exact index order"),
        "seed": 20260418,
        "sampler_seed": 33,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_VECTOR,
        "classes": 4,
        "index_pattern": "shuffled",
        "shuffle": True,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 16},
            "full": {"samples": 4096, "batch": 512},
            "profile": {"samples": 16384, "batch": 2048},
        },
        "setup": ("the dataset, its host snapshot, and the shuffled index "
                  "list are built once, outside the timer"),
        "timed": "exactly one feature_batch call",
        "cleanup": ("the returned NativeTensor is closed explicitly after "
                    "every repetition, outside the timer"),
        "notes": ("The shape a shuffled epoch really produces: the same "
                  "element count as feature_batch_large, read from scattered "
                  "rows rather than adjacent ones."),
    },
    "feature_batch_image": {
        "workload": HOST_TO_NATIVE,
        "label": "feature_batch_image",
        "operation": ("dataset.feature_batch(indices) over a rank-3 "
                      "per-sample geometry (1, 8, 8)"),
        "build": build_feature_batch,
        "gate": GATE_FEATURE_BATCH,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the native case allocates and transfers "
                             "and a NumPy gather does not"),
        "ratio_meaning": None,
        "correctness_reference": ("raw IEEE-754 bit equality with an "
                                  "independently gathered host array of the "
                                  "same rank-4 batch shape"),
        "seed": 20260419,
        "sampler_seed": 34,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_IMAGE,
        "classes": 3,
        "index_pattern": "sequential",
        "shuffle": False,
        "drop_last": False,
        "cache_state": None,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 64, "batch": 8},
            "full": {"samples": 2048, "batch": 128},
            "profile": {"samples": 8192, "batch": 512},
        },
        "setup": ("the dataset, its host snapshot, and the index list are "
                  "built once, outside the timer"),
        "timed": "exactly one feature_batch call",
        "cleanup": ("the returned NativeTensor is closed explicitly after "
                    "every repetition, outside the timer"),
        "notes": ("The second dataset geometry: many values per sample, and "
                  "a rank-4 batch, which is what a convolutional model "
                  "consumes."),
    },
    # -- loader delivery (a composition, reported apart) --------------------
    "loader_next_batch": {
        "workload": LOADER_DELIVERY,
        "label": "loader_next_batch",
        "operation": ("one successful next(iterator): the whole §9.4 "
                      "five-phase batch-delivery transaction"),
        "build": build_loader_delivery,
        "gate": GATE_DELIVERY,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — there is no reference implementation of "
                             "a transactional batch handoff, and no part of "
                             "this composition is isolated here"),
        "ratio_meaning": None,
        "correctness_reference": ("the candidate indices recorded before "
                                  "delivery, the delivered features in raw "
                                  "IEEE-754 bits against an independently "
                                  "gathered host array, the delivered "
                                  "targets by exact int64 equality, and the "
                                  "committed position advanced by exactly "
                                  "one batch"),
        "seed": 20260420,
        "sampler_seed": 35,
        "dtypes": DTYPES,
        "feature_shape": _GEOMETRY_IMAGE,
        "classes": 3,
        "index_pattern": None,
        "shuffle": True,
        "drop_last": False,
        "cache_state": WARM,
        "fixed_configuration": False,
        "configurations": {
            "smoke": {"samples": 128, "batch_size": 16},
            "full": {"samples": 2048, "batch_size": 64},
            "profile": {"samples": 8192, "batch_size": 128},
        },
        "setup": ("the canonical loader position is restored through the "
                  "public state loader, the epoch's order is warmed, and a "
                  "fresh one-epoch iterator is opened — all three per "
                  "repetition and all three outside the timer"),
        "timed": ("exactly one next(iterator): claim, construct, publish, "
                  "commit, and deliver"),
        "cleanup": ("the delivered feature tensor and the iterator are both "
                    "closed after every repetition, outside the timer; the "
                    "delivered target array is host memory and needs none"),
        "notes": ("A composition, and labelled one. It never substitutes for "
                  "the four isolated layers above: a single end-to-end "
                  "number cannot say which of them dominates, and that is "
                  "the only question §22.3 would ever reopen the export "
                  "count for. The epoch order is warmed in setup on "
                  "purpose — a state load invalidates the permutation "
                  "cache, so an unwarmed repetition would time an epoch's "
                  "permutation construction under a delivery label."),
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


# ===========================================================================
# Environment metadata
# ===========================================================================


def thread_environment():
    """The BLAS/threading environment variables that are **actually set**.

    Recorded so a reader can tell what the NumPy reference column ran
    under. Absent variables are simply not listed; nothing is invented,
    and no other environment variable is read or reported."""
    return {name: os.environ[name] for name in THREAD_ENVIRONMENT_VARIABLES
            if name in os.environ}


def environment():
    """Real introspection, and nothing identifying.

    No repository path, no working directory, no user name, no home
    directory, no host name, and no full environment dump: a
    characterization payload is meant to be readable by someone else, and
    none of those would help them. The backend half comes from the public
    ``cpp.backend_info()`` rather than from a hand-maintained restatement
    of it."""
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
        "setup_outside_timer": ("datasets, samplers, loaders, index sets, "
                                "iterators, restored positions, and cache "
                                "warming all happen outside the measured "
                                "region"),
        "cleanup_outside_timer": ("every native tensor is closed explicitly "
                                  "outside the measured region; nothing "
                                  "relies on garbage collection"),
        "no_threshold": ("no duration, throughput, ratio, memory, or "
                         "comparison threshold exists anywhere, and no CI "
                         "job fails on a number produced here"),
        "dtype_separation": ("float32 and float64 are measured, gated, and "
                             "reported separately and are never divided by "
                             "one another or ranked"),
        "ratio_rule": ("a ratio is published only for a case whose reference "
                       "is an independently written implementation of the "
                       "same operation at the same dtype over the same "
                       "inputs; a native_only case publishes none"),
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
    reference = None
    try:
        correctness = case.check()
        samples = measure(case.prepare, case.run, case.cleanup, warmup,
                          repetitions)
        if case.has_reference:
            reference = summarize(measure(
                case.reference_prepare, case.reference_run,
                case.reference_cleanup, warmup, repetitions))
    finally:
        case.teardown()
    statistics_block = summarize(samples)
    ratio = None
    if reference is not None and reference["median_ns"] > 0:
        ratio = statistics_block["median_ns"] / reference["median_ns"]
    return {
        "case": name,
        "workload": spec["workload"],
        "operation": spec["operation"],
        "dtype": dtype,
        "configuration": variant,
        "config": {key: int(value) for key, value in config.items()},
        "feature_shape": [int(dimension)
                          for dimension in spec["feature_shape"]],
        "seed": spec["seed"],
        "reference_type": spec["reference_type"],
        "native_only": spec["native_only"],
        "reference_detail": spec["reference_detail"],
        "cache_state": spec["cache_state"],
        "correctness": dict(passed=True, **correctness),
        "timed_layer": spec["timed"],
        "setup": spec["setup"],
        "cleanup": spec["cleanup"],
        "warmup": warmup,
        "statistics": statistics_block,
        "reference": reference,
        "ratio_to_reference": ratio,
        "ratio_meaning": spec["ratio_meaning"],
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
    built.
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
    chosen_dtypes = _resolve(dtypes, DTYPES, "dtype")
    variant = "smoke" if smoke else ("profile" if profile else "full")

    rows = []
    # Deterministic alternation: every case runs at each dtype in registry
    # order, case by case, so a slow drift in machine state touches both
    # widths alike instead of loading one of them.
    for name in selected:
        for dtype in chosen_dtypes:
            if dtype not in CASES[name]["dtypes"]:
                continue
            rows.append(run_case(name, dtype, warmup, repetitions, variant))
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

    Every row names its dtype, its median, and its spread. A
    ``native_only`` row shows no ratio, because there is none. The two
    dtypes are printed in separate sections and are never compared.
    """
    lines = []
    add = lines.append
    add(f"TensorForge native data-pipeline characterization "
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
            f"({backend['dtype']}/{backend['device']}, "
            f"available={backend['available']})")
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

    for dtype in payload["dtypes"]:
        rows = [row for row in payload["cases"] if row["dtype"] == dtype]
        if not rows:
            continue
        add(f"dtype: {dtype}")
        add("-" * 78)
        header = (f"  {'case':<32}{'median':>12}{'IQR':>12}{'rel':>8}"
                  f"{'ratio':>9}  {'reference':<12} gate")
        add(header)
        current = None
        for row in rows:
            if row["workload"] != current:
                current = row["workload"]
                add(f"  [{current}]")
            stats = row["statistics"]
            relative = stats["relative_iqr"]
            ratio = row["ratio_to_reference"]
            add(f"  {row['case']:<32}"
                f"{format_duration(stats['median_ns']):>12}"
                f"{format_duration(stats['iqr_ns']):>12}"
                f"{(f'{relative * 100:.1f}%' if relative is not None else 'n/a'):>8}"
                f"{(f'{ratio:.2f}x' if ratio is not None else 'none'):>9}  "
                f"{row['reference_type']:<12}"
                f"{row['correctness']['gate']}")
        add("")

    add("Reading this table")
    add("-" * 78)
    add("  Every row is one dtype's own characterization. There is no")
    add("  float32/float64 ratio anywhere in this output and none is")
    add("  implied: that number is a property of one machine's memory")
    add("  bandwidth, not of TensorForge, and publishing it would turn a")
    add("  measurement into a promise the project cannot keep.")
    add("")
    add("  'ratio' is this case's median divided by the median of an")
    add("  independently written implementation of the same operation at")
    add("  the same dtype over the same inputs. A native_only row shows")
    add("  'none', because no honest equivalent exists to divide by; its")
    add("  correctness gate is still real. Each case's reference_detail and")
    add("  ratio_meaning in --json say exactly what was compared.")
    add("")
    add("  The cold and warm permutation cases are separate rows on")
    add("  purpose and are never averaged: one measures constructing an")
    add("  epoch's order, the other measures reading the cached one.")
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Characterize the deterministic native data pipeline: "
                     "dataset indexing, batch planning, permutation "
                     "construction, and host-to-native materialization "
                     "(measurement only; no speed is asserted, no threshold "
                     "exists, and no result file is written)."))
    parser.add_argument("--case", choices=tuple(CASES), default=None,
                        help="run a single case (default: all)")
    parser.add_argument("--workload", choices=WORKLOADS, default=None,
                        help="run one workload family (default: all)")
    parser.add_argument("--dtype", choices=DTYPES, action="append",
                        default=None,
                        help="measure only this dtype (repeatable)")
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
