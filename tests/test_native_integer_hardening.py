"""Phase K, milestone K7 — adversarial hardening of the native integer
and indexing stack.

K1 through K4 each shipped a layer and proved it; K5 proved the
compatibility boundary and K6 proved the integration. **K7 attacks all of
it at once** and proves one sentence:

    **A rejected or failed integer operation leaves the entire observable
    world exactly as it found it — no published output, no partial write,
    no leaked storage, no moved registry, and no stale native error — and
    it does so under a ``BaseException`` exactly as it does under a
    ``ValueError``.**

What this module owns (design §30.1's "adversarial hardening" row), and
why none of it is a duplicate of K3's or K4's own failure section:

1. **§27's four injection families at every *actual* allocating path**,
   resolved from the live call graph rather than assumed. The families are
   host validation/normalization, native allocation **through the
   backend's own thread-local arm**, host→native transfer/materialization,
   and kernel execution — and a family that genuinely does not apply to a
   path is recorded as ``N/A`` with its technical reason in
   ``INJECTION_MATRIX`` rather than being faked with a representative
   injection borrowed from a neighbour. K3 and K4 injected with
   monkeypatched stand-ins; **every allocation row here fires the real
   ``tf_test_arm_alloc_failure`` countdown**, armed at the exact
   production seam it targets. Where one export is reached from two
   different call sites the matrix carries **two rows**, not one:
   ``index_select`` materializes through ``tf_core_contiguous_copy`` once
   for its floating source and once for its ``int64`` index, so the second
   is driven by a call journal that delegates the first to the real export
   and fails only the second — an injection that fails immediately can
   never reach it, and one representative failure may not stand in for
   both.
2. **One reusable before/after fingerprint of the observable world** around
   every rejection and every injected failure — both operands, an
   unrelated parameter with its version and gradient, a persistent and a
   non-persistent buffer, a live optimizer, a registered generator, every
   capability registry and dtype table, ``experimental.__all__``, both
   global RNGs, the environment, a watched directory, the live-storage
   count, and the native error slot. Every component has a perturbation
   control proving it can notice the change it exists for, and
   ``test_every_injection_matrix_owner_carries_the_complete_world_fingerprint``
   makes "every injected failure" literal by reading this module's own AST
   rather than leaving it as prose. Where a rejection needs a deliberate
   instrument — an emptied ``INDEX_DTYPES``, a lowered ``_INT64_MAX`` — the
   instrument is applied **first** and the fingerprint taken after it, so
   what is proved unchanged is what the rejected call touches.
3. **Retained-reference cleanup proofs.** A normal exception can make
   broken cleanup *look* correct, because reference counting frees the
   object during unwinding. Every publication-failure test therefore keeps
   the allocated core or storage alive in an external list and asserts it
   was closed by production cleanup **while the reference still exists**.
   No assertion here depends on ``__del__`` or on collection timing.
4. **A ``BaseException`` through every cleanup-capable Python seam**, so
   the unconditional ``finally``/``except BaseException`` blocks are
   proved unconditional rather than proved against the easy case.
5. **The malformed-metadata *and* dtype-role matrices for both exports,
   separately.** ``tf_core_argmax`` and ``tf_core_index_select`` have
   *different* validation lists (§22.10), so they get different matrices
   and no blanket helper obscures which rule each one owns. Every
   rejection — metadata or role — prefills every operand with distinctive
   values, asserts not one byte moved in **any** of them (two handles for
   argmax, three for index_select), allocates no native storage, and
   leaves the error slot clean after Python's ``errcheck`` hook has run; a
   valid control proves the detector can see a single-element write and
   that the permitted role combination really executes.
6. **The complete index scan before any write**, driven at the C ABI with
   an index vector whose invalid value comes *after* several valid ones.

Discipline (integer design §29.6, §30.2), and nothing here relaxes it:

* **Exact equality only** for integers — Python ``int``, ``np.int64``
  ``array_equal``, or raw bytes — and **raw IEEE-754 bits** for floating
  values, ``uint32`` for float32 and ``uint64`` for float64, each dtype
  compared only against itself. No tolerance, no ``allclose``, no
  ``approx``, and no cross-dtype numeric equality appears in this file.
* The ``live_storages`` tracker installs itself **outside**
  ``monkeypatch``, and a test proves a mid-test ``monkeypatch.undo()``
  cannot disarm it.
* Every injector has a non-vacuity control proving it fired, at the
  intended seam, the intended number of times.
* Every parser and scanner has a planted **in-memory** negative control.
  No repository file is mutated to plant one.
* **No test starts a thread**, and a scan of this module's own AST proves
  it imports and names no concurrency machinery (§25).
* No test asserts a complete error message, a timing, a benchmark number,
  a garbage-collection event, or a total suite count.

**K7 adds no production code.** Every seam it uses already exists: the
backend's thread-local allocation arm, the private ``_typed`` allocators,
``NativeTensor._from_core``, ``cpp._exact_host_array``, the ctypes
declarations, and the error-slot API. No production fault hook, no public
failure API, no allocation counter, and no test-only export was added.

Selector: python -m pytest -q tests/test_native_integer_hardening.py
"""
import ast
import contextlib
import gc
import inspect
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tensorforge
import tensorforge.experimental as experimental
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeGenerator,
    NativeLinear,
    NativeModule,
    NativeParameter,
    NativeReLU,
    NativeTensor,
)
from tensorforge.experimental import native_checkpoint as checkpoint_module
from tensorforge.experimental import native_data_loader as loader_module
from tensorforge.experimental import native_optimizer_state as optimizer_state
from tensorforge.experimental import native_sampler as sampler_module

REPO_ROOT = Path(__file__).resolve().parent.parent
THIS_MODULE = Path(__file__).resolve()

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)
needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="the build has no deterministic allocation-failure arm",
)

# Written here independently of the modules under test, so a silent change
# fails rather than propagating.
FLOATING_DTYPES = ("float64", "float32")
INDEX_DTYPE = "int64"
BIT_VIEW = {"float64": np.uint64, "float32": np.uint32}
NUMPY_DTYPE = {"float64": np.float64, "float32": np.float32}

# The inventories K7 must leave exactly where K6 left them.
K7_EXPORT_COUNT = 56
K7_CTEST_COUNT = 27
K7_EXAMPLE_COUNT = 17
# 9 when K7 landed; 10 since **K8** added exactly one benchmark
# (benchmarks/benchmark_native_integer.py). The number is updated rather
# than the assertion relaxed: K7's own benchmark delta is still zero, K8's
# artifact is named and subtracted below, and an unrecorded addition still
# fails an exact equality.
K7_OWN_BENCHMARK_COUNT = 9
POST_K7_BENCHMARKS = {"benchmark_native_integer.py": "K8"}
K7_BENCHMARK_COUNT = K7_OWN_BENCHMARK_COUNT + len(POST_K7_BENCHMARKS)
K7_EXPERIMENTAL_EXPORTS = 25
K7_CHECKED_KERNELS = 38

# The two exports the phase added. K8's two artifacts landed after K7 and
# are therefore named as **present** rather than absent — the entry moved
# instead of being deleted, so this stays a claim about the ladder — while
# K9's closure module must still be absent.
INDEXING_EXPORTS = ("tf_core_argmax", "tf_core_index_select")
K8_ARTIFACTS = (
    "benchmarks/benchmark_native_integer.py",
    "tests/test_native_integer_benchmark.py",
)
K9_ARTIFACTS = (
    "tests/test_native_phase_k_closure.py",
)

# Concurrency machinery §25 forbids this module from touching at all.
CONCURRENCY_NAMES = (
    "threading", "multiprocessing", "concurrent", "asyncio", "Thread",
    "Lock", "RLock", "Semaphore", "Condition", "Queue", "Future",
    "ThreadPoolExecutor", "ProcessPoolExecutor", "start_new_thread",
)


# ===========================================================================
# 0. Fixtures and instruments
# ===========================================================================

@pytest.fixture(autouse=True)
def _disarm_allocation_faults():
    """No injected allocation failure and no recorded native error survives
    a test, whatever it did.

    Disarmed here **as well as** in every arming test's own ``finally``, so
    a test that dies between the two cannot poison the next one (§27.2).
    """
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages():
    """The ids of every ``NativeStorage`` currently open.

    Installed with an explicit save/restore rather than through
    ``monkeypatch`` (§30.2): several tests here take their own injection
    back out mid-test, and a tracker installed through the same
    ``monkeypatch`` would be silently uninstalled with it — leaving every
    later ``close()`` unrecorded and every live-storage assertion vacuous.
    """
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
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


def settled(live):
    """The live-storage count after a collection.

    Collection *settles* the count; it is never the proof that anything was
    released — every test here closes what it owns explicitly first, and the
    retained-reference tests assert closure while still holding a strong
    reference.
    """
    gc.collect()
    return len(live)


@contextlib.contextmanager
def no_native_allocation():
    """Assert the block allocates **no** native storage at all.

    Strictly stronger than a baseline check, and it is the instrument
    §13.11 needs: "nothing is allocated before validation succeeds" is a
    different claim from "whatever was allocated was released".
    """
    allocated = []
    original_init = cpp.NativeStorage.__init__

    def counting_init(self, *args, **kwargs):
        allocated.append((args, tuple(sorted(kwargs))))
        original_init(self, *args, **kwargs)

    cpp.NativeStorage.__init__ = counting_init
    try:
        yield allocated
        assert not allocated, (
            f"{len(allocated)} native storages were allocated by a call that "
            f"must allocate nothing: {allocated}"
        )
    finally:
        cpp.NativeStorage.__init__ = original_init


@contextlib.contextmanager
def arm_native_allocation(predicate, occurrence=1):
    """Fire the **backend's own** thread-local allocation-failure arm at
    exactly the storage allocation ``predicate`` selects.

    The failure itself is entirely native: ``tf::should_fail_alloc`` throws
    a real ``std::bad_alloc`` inside ``tf_storage_create_typed``, the guard
    records ``TF_ERROR_ALLOC``, and the errcheck hook raises
    ``MemoryError``. Nothing here stands in for that. The Python wrapper
    decides only **when** to arm, which is what makes the position exact:
    the countdown is set to 1 immediately before the targeted
    ``NativeStorage.__init__`` runs, and nothing else can allocate between
    the two — so no ordinal counting of unrelated allocations is involved
    and no other call site can absorb the injection.

    Yields the list of allocations it armed on, which is the non-vacuity
    record: a test asserts the seam was reached with the size and dtype it
    meant to hit.
    """
    armed = []
    matches = []
    original_init = cpp.NativeStorage.__init__

    def arming_init(self, size, dtype=None, device="cpu", **kwargs):
        if not armed and predicate(int(size), dtype):
            matches.append((int(size), dtype))
            if len(matches) == occurrence:
                armed.append((int(size), dtype))
                cpp._arm_alloc_failure(1)
        original_init(self, size, dtype=dtype, device=device, **kwargs)

    cpp.NativeStorage.__init__ = arming_init
    try:
        yield armed
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


class Boom(Exception):
    """The one injected ``Exception``, so a test can never mistake an
    accidental production error for its own injection."""


class Abort(BaseException):
    """A ``BaseException`` that is deliberately **not** an ``Exception``, so
    an unconditional cleanup is proved unconditional rather than proved
    against the easy case (§27.1)."""


# --- exact comparison helpers ---------------------------------------------

def bits(array):
    """One array's raw IEEE-754 bit patterns, as unsigned integers.

    The **only** way this module compares a floating value. ``==`` calls
    two NaNs unequal, calls ``+0.0`` and ``-0.0`` equal, and cannot see a
    NaN payload at all, so it can prove none of what a bit-preserving copy
    promises (§29.6). The dtype is asserted rather than coerced: a helper
    that quietly converted could report a match that existed only after a
    conversion this runtime does not perform.
    """
    array = np.ascontiguousarray(array)
    name = str(array.dtype)
    assert name in BIT_VIEW, f"{name} is not a floating dtype"
    return array.view(BIT_VIEW[name]).tolist()


def same_bits(left, right):
    return (np.shape(left) == np.shape(right)
            and bits(left) == bits(right))


def exact_ints(array):
    """One integer array's values, exactly. Never a float conversion."""
    array = np.ascontiguousarray(array)
    assert array.dtype == np.dtype(np.int64), array.dtype
    return array.tolist()


def floating(values, dtype):
    return NativeTensor.from_array(np.asarray(values, dtype=np.float64),
                                   dtype=dtype)


def index(values):
    return NativeTensor.from_int64_array(np.asarray(values, dtype=np.int64))


def typed_core(shape, dtype, values=None):
    """A ``NativeTensorCore`` at exactly ``dtype``, optionally prefilled.

    Built through the private typed allocator because the direct C ABI
    matrix below needs an ``int64`` destination, which no public
    constructor produces — that is the point of §5.5, not a gap.
    """
    core = cpp.NativeTensorCore._typed(shape, dtype)
    if values is not None:
        core._storage.copy_from(
            np.ascontiguousarray(values, dtype=cpp._DTYPE_NUMPY[dtype]))
    return core


def raw_bytes(core):
    """A core's storage contents as raw bytes — the no-write instrument."""
    return core.to_numpy().tobytes()


def strided_view(base):
    """A genuinely **non-contiguous** rank-1 view over ``base``'s storage,
    taking every second element.

    Built through the public view constructor at the Core layer because
    none of the shipped view ops can produce one at rank 1: ``reshape``
    refuses a non-contiguous input, ``transpose`` is the identity, and
    ``narrow`` preserves its parent's strides. The view **borrows** —
    closing it leaves ``base`` open.
    """
    core = base._require_open()
    storage = core.storage
    length = core.numel // 2
    view = cpp.NativeTensorView(storage, (length,), strides=(2,), offset=0)
    return NativeTensor._from_core(
        cpp.NativeTensorCore(storage, view, owns_storage=False))


# ===========================================================================
# 1. The observable-world fingerprint
# ===========================================================================
#
# One reusable snapshot, in semantic pieces rather than one opaque blob, so
# a failure names the component that moved and a harmless internal
# reformatting does not force a rewrite of the matrix.

class Sentinels:
    """Unrelated native state a Phase-K rejection may never touch: a
    registered parameter with a real gradient and a moved version, a
    **persistent** buffer, a **non-persistent** buffer, a live optimizer
    holding Adam moments for that parameter, and a registered generator
    that has actually drawn.

    Held together only so a test can build one and close it explicitly. No
    production analogue exists or is implied.
    """

    __slots__ = ("model", "optimizer", "generator", "parameter")

    def __init__(self):
        self.model = SentinelModel()
        self.generator = self.model.rng
        self.parameter = self.model.linear.weight
        self.optimizer = NativeAdam(self.model.parameters(), lr=0.01)
        features = NativeTensor.from_array(
            np.linspace(-1.0, 1.0, 12).reshape(3, 4))
        try:
            out = self.model(features)
            loss = out.sum()
            loss.backward()
            self.optimizer.step()
            loss.close()
            out.close()
        finally:
            features.close()

    def close(self):
        """Explicit cleanup, in the established order: the optimizer's
        moments, then every unique gradient, parameter, and buffer.

        The gradients are closed **explicitly** and are not left to
        ``close()`` on the parameter: a gradient is a separate owning
        tensor that ``NativeTensor.close`` deliberately does not reach, so
        dropping it would make this helper's teardown depend on ``__del__``
        timing — exactly what §9 forbids every assertion in this module to
        rest on. Nothing here relies on garbage collection.
        """
        self.optimizer.close()
        seen = set()
        parameters = list(self.model.named_parameters())
        for _, parameter in parameters:
            gradient = None if parameter is None else parameter.grad
            if gradient is not None and id(gradient) not in seen:
                seen.add(id(gradient))
                gradient.close()
        for _, tensor in parameters + list(self.model.named_buffers()):
            if tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                tensor.close()


class SentinelModel(NativeModule):
    """Trainable parameters, both buffer kinds, and a registered generator
    — one object covering every family a rejection must not disturb."""

    def __init__(self):
        super().__init__()
        self.linear = NativeLinear(4, 3, seed=11)
        self.relu = NativeReLU()
        self.register_buffer(
            "kept", NativeTensor.from_array(np.array([1.5, -2.5, 0.25])),
            persistent=True)
        self.register_buffer(
            "scratch", NativeTensor.from_array(np.array([4.0, 8.0])),
            persistent=False)
        self.rng = NativeGenerator(seed=909)
        self.register_generator("rng", self.rng)

    def forward(self, x):
        return self.relu(self.linear(x))


def tensor_view(tensor):
    """Everything a rejection must leave alone about one operand.

    A closed tensor has no readable metadata, so the closed flag is read
    first and the rest is skipped: a fingerprint that *raised* after a close
    would be an instrument unable to report the very change it exists to
    notice. Integer payloads are compared as exact bytes, floating ones as
    raw IEEE-754 bit patterns — never as numbers.
    """
    if tensor is None:
        return None
    closed = tensor.closed
    if closed:
        return {"id": id(tensor), "closed": True}
    grad = tensor.grad
    return {
        "id": id(tensor),
        "closed": False,
        "core_id": id(tensor._core),
        "storage_id": id(tensor._core.storage),
        "dtype": tensor.dtype,
        "device": tensor.device,
        "shape": tensor.shape,
        "strides": tensor.strides,
        "offset": tensor._core.offset,
        "numel": tensor.numel,
        "ndim": tensor.ndim,
        "contiguous": tensor.contiguous,
        "owns_core": tensor._owns_core,
        "requires_grad": tensor.requires_grad,
        "is_leaf": tensor.is_leaf,
        "op": tensor._op,
        "parents": tuple(id(parent) for parent in tensor._parents),
        "has_backward": tensor._backward is not None,
        "graph_resources": len(tensor._graph_resources),
        # Raw bytes at every dtype: exact for int64, bit-exact for floating.
        "payload": tensor.to_numpy().tobytes(),
        "grad_id": None if grad is None else id(grad),
        "grad_payload": (None if grad is None
                         else grad.to_numpy().tobytes()),
    }


def sentinel_view(sentinels):
    """Every unrelated-state family, by identity, value, and version."""
    if sentinels is None:
        return None
    model = sentinels.model
    parameters = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        parameters[name] = (
            id(parameter),
            parameter.to_numpy().tobytes(),
            parameter.version,
            parameter.requires_grad,
            None if grad is None else grad.to_numpy().tobytes(),
        )
    buffers = {}
    for name, buffer in model.named_buffers():
        buffers[name] = (None if buffer is None
                         else (id(buffer), buffer.to_numpy().tobytes()))
    return {
        "parameters": parameters,
        "buffers": buffers,
        "persistent": tuple(sorted(
            name for name, _ in model._persistent_named_buffers())),
        "generator": sentinels.generator.state(),
        "generator_calls": sentinels.generator.calls,
        "optimizer_parameters": tuple(
            id(parameter) for parameter in sentinels.optimizer.parameters()),
        "optimizer": json.dumps(sentinels.optimizer.state_dict(),
                                sort_keys=True, default=repr),
        "training": model.training,
    }


def registry_view():
    """Every capability registry, dtype table, operation inventory, format,
    and version a Phase-K failure may not move."""
    return {
        "supported_dtypes": cpp.SUPPORTED_DTYPES,
        "index_dtypes": cpp.INDEX_DTYPES,
        "supported_devices": cpp.SUPPORTED_DEVICES,
        "unsupported": cpp.UNSUPPORTED,
        "raw_kernel_dtypes": cpp.RAW_KERNEL_DTYPES,
        "default_dtype": cpp.normalize_dtype(None),
        "dtype_codes": tuple(sorted(cpp._DTYPE_CODES.items())),
        "item_sizes": tuple(sorted(cpp._DTYPE_ITEM_SIZES.items())),
        "numpy_dtypes": tuple(sorted(
            (name, np.dtype(value).name)
            for name, value in cpp._DTYPE_NUMPY.items())),
        "host_arrays": tuple(sorted(cpp._CHECKED_HOST_ARRAYS)),
        "raw_kernels": cpp.RAW_KERNELS,
        "tensor_core_kernels": cpp.TENSOR_CORE_KERNELS,
        "tensor_core_ops": cpp.TENSOR_CORE_OPS,
        "autograd_ops": cpp.AUTOGRAD_OPS,
        "checked_kernels": cpp._CHECKED_KERNELS,
        "backend_info": json.dumps(cpp.backend_info(), sort_keys=True,
                                   default=repr),
        "checkpoint_format": checkpoint_module._FORMAT,
        "checkpoint_version": checkpoint_module._FORMAT_VERSION,
        "checkpoint_versions": checkpoint_module._SUPPORTED_FORMAT_VERSIONS,
        "optimizer_state_version": optimizer_state.FORMAT_VERSION,
        "loader_format": loader_module._FORMAT,
        "loader_version": loader_module._FORMAT_VERSION,
        "loader_versions": loader_module._SUPPORTED_FORMAT_VERSIONS,
        "sampler_format": sampler_module._FORMAT,
        "sampler_version": sampler_module._FORMAT_VERSION,
        "sampler_versions": sampler_module._SUPPORTED_FORMAT_VERSIONS,
        "experimental_all": tuple(experimental.__all__),
        "stable_all": tuple(tensorforge.__all__),
    }


def globals_view(directory=None):
    """The process-level state a Phase-K operation never touches.

    Deliberately **not** a claim of complete process purity: it names the
    globals the design actually says the phase does not use. Collection
    timing, allocator internals, and unrelated object ids are excluded on
    purpose — they are not contracts.
    """
    numpy_state = np.random.get_state()
    view = {
        "python_random": random.getstate(),
        "numpy_random": (numpy_state[0], numpy_state[1].tolist(),
                         numpy_state[2], numpy_state[3], numpy_state[4]),
        "environ": dict(os.environ),
        "cwd": os.getcwd(),
        "registries": registry_view(),
    }
    if directory is not None:
        view["files"] = sorted(
            (str(path.relative_to(directory)), path.stat().st_size)
            for path in Path(directory).rglob("*") if path.is_file()
        )
    return view


def error_slot():
    """The calling thread's native error code — a component in its own
    right, because a rejection that leaves a stale code behind can change
    a *later* call's reported outcome."""
    if not cpp.is_available():
        return None
    return cpp._require_library().tf_last_error_code()


def world(*, operands=(), sentinels=None, directory=None, live=None):
    """One comparable snapshot of everything a rejected or failed integer
    operation must leave untouched."""
    return {
        "operands": tuple(tensor_view(operand) for operand in operands),
        "sentinels": sentinel_view(sentinels),
        "globals": globals_view(directory),
        "error_slot": error_slot(),
        "live_storages": None if live is None else settled(live),
    }


@contextlib.contextmanager
def unchanged_world(*, operands=(), sentinels=None, directory=None,
                    live=None):
    """Snapshot, hand over, and assert the whole fingerprint is identical
    afterwards."""
    before = world(operands=operands, sentinels=sentinels,
                   directory=directory, live=live)
    yield before
    after = world(operands=operands, sentinels=sentinels,
                  directory=directory, live=live)
    for key in before:
        assert after[key] == before[key], f"the {key} component moved"
    assert after == before


# ===========================================================================
# 2. The instruments themselves — non-vacuity controls
# ===========================================================================
#
# Every equality below is evidence only if the fingerprint can actually
# change, so each component is driven against the mutation it exists to
# notice. Every perturbed global or registry is restored in a ``finally``.

@needs_native
def test_the_live_storage_tracker_survives_a_monkeypatch_undo(monkeypatch,
                                                              live_storages):
    """The control every live-storage assertion in this module rests on: a
    mid-test ``monkeypatch.undo()`` must not disarm the tracker."""
    monkeypatch.setattr(loader_module, "_FORMAT_VERSION", 1)
    monkeypatch.undo()
    baseline = settled(live_storages)
    held = index(np.array([1, 2, 3], dtype=np.int64))
    assert settled(live_storages) == baseline + 1
    held.close()
    assert settled(live_storages) == baseline


@needs_native
def test_the_live_storage_tracker_notices_a_deliberately_retained_tensor(
        live_storages):
    """The tracker's own negative control: an open tensor it is *not* told
    about must show up as a leak, so "returned to baseline" means
    something."""
    baseline = settled(live_storages)
    retained = index(np.array([7, 8], dtype=np.int64))
    try:
        assert settled(live_storages) == baseline + 1, (
            "the tracker did not notice a retained open tensor"
        )
    finally:
        retained.close()
    assert settled(live_storages) == baseline


@needs_native
def test_the_no_allocation_probe_can_actually_fail():
    with pytest.raises(AssertionError):
        with no_native_allocation():
            index(np.array([1], dtype=np.int64)).close()


@needs_native
def test_the_bit_and_integer_comparisons_can_actually_fail():
    """Raw bits separate values a tolerance calls equal, at both widths;
    the integer helper is exact and never converts."""
    for dtype in FLOATING_DTYPES:
        zero = np.array([0.0], dtype=NUMPY_DTYPE[dtype])
        minus = np.array([-0.0], dtype=NUMPY_DTYPE[dtype])
        assert float(zero[0]) == float(minus[0])       # a number cannot see it
        assert bits(zero) != bits(minus)               # the bits can
        nan_a = np.array([np.nan], dtype=NUMPY_DTYPE[dtype])
        assert bits(nan_a) == bits(nan_a)              # equal to itself
    with pytest.raises(AssertionError):
        bits(np.array([1], dtype=np.int64))
    with pytest.raises(AssertionError):
        exact_ints(np.array([1.0], dtype=np.float64))
    assert exact_ints(np.array([2 ** 62, -(2 ** 62)], dtype=np.int64)) == \
        [2 ** 62, -(2 ** 62)]


@needs_native
def test_the_operand_component_notices_every_change_it_exists_for():
    """Every field of ``tensor_view``, driven against a real change."""
    source = floating([[1.0, 2.0], [3.0, 4.0]], "float64")
    indices = index(np.array([1, 0], dtype=np.int64))
    try:
        base = tensor_view(source)
        assert tensor_view(source) == base
        # value
        other = floating([[1.0, 2.0], [3.0, 5.0]], "float64")
        assert tensor_view(other) != base
        other.close()
        # dtype
        narrow = floating([[1.0, 2.0], [3.0, 4.0]], "float32")
        assert tensor_view(narrow)["dtype"] != base["dtype"]
        narrow.close()
        # shape / strides / contiguity / ownership
        transposed = source.T
        moved = tensor_view(transposed)
        assert moved["strides"] != base["strides"]
        assert moved["contiguous"] != base["contiguous"]
        assert moved["owns_core"] != base["owns_core"]
        transposed.close()
        # integer payload, by exact bytes
        index_base = tensor_view(indices)
        other_indices = index(np.array([1, 1], dtype=np.int64))
        assert tensor_view(other_indices)["payload"] != index_base["payload"]
        other_indices.close()
        # graph state and gradient
        tracked = floating([[1.0, 2.0], [3.0, 4.0]], "float64")
        tracked._init_requires_grad(True)
        assert tensor_view(tracked)["requires_grad"] is True
        assert tensor_view(tracked)["grad_payload"] is None
        total = tracked.sum()
        total.backward()
        with_grad = tensor_view(tracked)
        assert with_grad["grad_payload"] is not None
        assert with_grad["grad_id"] is not None
        total.close()
        tracked.grad.close()
        tracked.close()
        # closed state
        closing = index(np.array([5], dtype=np.int64))
        open_view = tensor_view(closing)
        closing.close()
        assert tensor_view(closing) != open_view
        assert tensor_view(closing)["closed"] is True
    finally:
        indices.close()
        source.close()


@needs_native
def test_the_sentinel_component_notices_every_family():
    """Parameter value and version, gradient, both buffer kinds, the
    optimizer's own state, and the generator must each be visible."""
    sentinels = Sentinels()
    try:
        base = sentinel_view(sentinels)
        assert sentinel_view(sentinels) == base
        # a parameter value replacement moves value **and** version
        parameter = sentinels.parameter
        replacement = NativeTensor.from_array(
            np.zeros(parameter.shape, dtype=np.float64))
        try:
            parameter.copy_value_(replacement)
        finally:
            replacement.close()
        moved = sentinel_view(sentinels)
        assert moved != base
        assert moved["parameters"]["linear.weight"][2] != \
            base["parameters"]["linear.weight"][2]
        # both buffer kinds are present and each is visible by value
        assert set(moved["buffers"]) >= {"kept", "scratch"}
        assert moved["persistent"] == ("kept",), moved["persistent"]
        for name in ("kept", "scratch"):
            buffer = dict(sentinels.model.named_buffers())[name]
            before = sentinel_view(sentinels)["buffers"][name]
            buffer._core.storage.copy_from(
                np.full(buffer.numel, 3.0, dtype=np.float64))
            assert sentinel_view(sentinels)["buffers"][name] != before, name
        # the generator's own state
        before_generator = sentinel_view(sentinels)["generator"]
        sentinels.generator.reseed(4242)
        assert sentinel_view(sentinels)["generator"] != before_generator
        # the optimizer's counters
        before_optimizer = sentinel_view(sentinels)["optimizer"]
        sentinels.optimizer.step()
        assert sentinel_view(sentinels)["optimizer"] != before_optimizer
    finally:
        sentinels.close()


def test_the_registry_component_notices_a_moved_registry(monkeypatch):
    base = registry_view()
    assert registry_view() == base
    monkeypatch.setattr(cpp, "INDEX_DTYPES", ("int64", "int32"))
    assert registry_view() != base
    monkeypatch.undo()
    assert registry_view() == base
    monkeypatch.setattr(loader_module, "_FORMAT_VERSION", 2)
    assert registry_view() != base
    monkeypatch.undo()
    assert registry_view() == base


def test_the_globals_component_notices_an_rng_move_and_a_written_file(
        tmp_path):
    base = globals_view(directory=tmp_path)
    assert base["files"] == []
    assert globals_view(directory=tmp_path) == base
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.random()
        assert globals_view(directory=tmp_path) != base
        random.setstate(python_state)
        assert globals_view(directory=tmp_path) == base
        np.random.random()
        assert globals_view(directory=tmp_path) != base
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    assert globals_view(directory=tmp_path) == base
    (tmp_path / "written.bin").write_bytes(b"x")
    assert globals_view(directory=tmp_path) != base
    assert globals_view(directory=tmp_path)["files"] == [("written.bin", 1)]


@needs_native
def test_the_error_slot_component_notices_a_recorded_error():
    """The slot is part of the world, so it needs its own control."""
    library = cpp._require_library()
    library.tf_clear_error()
    assert error_slot() == cpp.TF_OK
    hook = library.tf_core_argmax.errcheck
    source = typed_core((4,), "float64", [1.0, 4.0, 2.0, 3.0])
    destination = typed_core((1,), INDEX_DTYPE)
    try:
        library.tf_core_argmax.errcheck = \
            lambda result, function, arguments: result
        library.tf_core_argmax(
            source._storage._require_open(), 0,
            destination._storage._require_open(), 0, 4, 1)
        assert error_slot() != cpp.TF_OK, "the slot component saw nothing"
    finally:
        library.tf_core_argmax.errcheck = hook
        library.tf_clear_error()
        destination.close()
        source.close()
    assert error_slot() == cpp.TF_OK


@needs_native
def test_the_live_storage_component_of_the_world_can_move(live_storages):
    """The last fingerprint component: the count itself."""
    base = world(live=live_storages)
    held = index(np.array([1, 2], dtype=np.int64))
    try:
        assert world(live=live_storages) != base
    finally:
        held.close()
    assert world(live=live_storages) == base


@needs_native
def test_the_unchanged_world_context_can_actually_fail(live_storages):
    """The context manager every rejection test uses, driven against a
    change it must catch."""
    source = floating([[1.0, 2.0]], "float64")
    try:
        with pytest.raises(AssertionError):
            with unchanged_world(operands=(source,), live=live_storages):
                leaked = index(np.array([1], dtype=np.int64))
                leaked.close()
                random.random()
    finally:
        source.close()


@needs_fault_injection
def test_the_allocation_arm_helper_fires_at_the_seam_it_names(live_storages):
    """The control for every allocation row below: the arm helper must fire
    on the allocation its predicate selects and on no other, and the very
    next unarmed call must succeed."""
    baseline = settled(live_storages)
    with arm_native_allocation(lambda size, dtype: dtype == INDEX_DTYPE) \
            as armed:
        # A floating allocation is not the target and must succeed.
        untouched = floating([[1.0, 2.0]], "float64")
        assert not armed, "the arm fired on a floating allocation"
        untouched.close()
        with pytest.raises(MemoryError):
            index(np.array([1, 2, 3], dtype=np.int64))
    assert armed == [(3, INDEX_DTYPE)], armed
    assert settled(live_storages) == baseline
    # ...and a valid construction succeeds once the arm is gone.
    recovered = index(np.array([1, 2, 3], dtype=np.int64))
    try:
        assert exact_ints(recovered.to_numpy()) == [1, 2, 3]
    finally:
        recovered.close()
    assert settled(live_storages) == baseline


# ===========================================================================
# 3. The injection path matrix
# ===========================================================================
#
# §27.2 requires four **distinct** injection families and forbids labelling
# one as another. It does not require that all four apply to every path, and
# pretending they do would be the same defect in the other direction — so
# every row below is either an injected position with a named owner test, or
# an ``N/A`` with the technical reason it cannot exist.
#
# Resolved from the live call graph, not assumed:
#
#   NativeTensor.from_int64_array
#     -> requires_grad type/value            (pre-allocation)
#     -> cpp._normalize_index_dtype          (pre-allocation registry gate)
#     -> NativeTensorCore._from_int64_array
#          -> cpp._exact_host_array          (host validation)
#          -> cpp._as_shape
#          -> NativeStorage._from_int64_array
#               -> cpp._exact_host_array     (host normalization, 2nd site)
#               -> NativeStorage._typed      (native allocation)
#               -> NativeStorage.copy_from   (host -> native transfer)
#          -> _contiguous_view / core        (core construction)
#     -> NativeTensor._from_core             (wrapper publication)

FAMILY_HOST = "host validation or normalization"
FAMILY_ALLOC = "native allocation (thread-local arm)"
FAMILY_TRANSFER = "host-to-native transfer or materialization"
FAMILY_KERNEL = "native kernel execution"

# (operation, position, family, owner test or the N/A reason)
INJECTION_MATRIX = (
    ("from_int64_array", "exact host array validation", FAMILY_HOST,
     "test_from_int64_array_host_validation_allocates_nothing"),
    ("from_int64_array", "index dtype registry gate", FAMILY_HOST,
     "test_from_int64_array_asks_the_index_registry_before_the_array"),
    ("from_int64_array", "int64 storage allocation", FAMILY_ALLOC,
     "test_from_int64_array_storage_allocation_failure_leaks_nothing"),
    ("from_int64_array", "host to storage copy", FAMILY_TRANSFER,
     "test_from_int64_array_transfer_failure_closes_the_storage"),
    ("from_int64_array", "core and view construction", FAMILY_TRANSFER,
     "test_from_int64_array_core_construction_failure_closes_the_storage"),
    ("from_int64_array", "public wrapper publication", FAMILY_TRANSFER,
     "test_from_int64_array_publication_failure_closes_the_core"),
    ("from_int64_array", "kernel execution", FAMILY_KERNEL,
     "N/A: construction runs no compute kernel at all — its only native "
     "calls are the storage creator and tf_storage_copy_from, and the "
     "second is the transfer row above rather than a second kernel row"),
    ("int64 contiguous_copy", "host validation", FAMILY_HOST,
     "N/A: the operand is already native, so no host buffer is validated "
     "or normalized on this path"),
    ("int64 contiguous_copy", "destination allocation", FAMILY_ALLOC,
     "test_int64_contiguous_copy_destination_allocation_failure"),
    ("int64 contiguous_copy", "host to native transfer", FAMILY_TRANSFER,
     "N/A: tf_core_contiguous_copy is a native storage-to-storage gather; "
     "no tensor data crosses the host boundary, only layout metadata"),
    ("int64 contiguous_copy", "tf_core_contiguous_copy", FAMILY_KERNEL,
     "test_int64_contiguous_copy_kernel_failure_closes_the_destination"),
    ("int64 contiguous_copy", "wrapper publication", FAMILY_KERNEL,
     "test_int64_contiguous_copy_publication_failure_closes_the_core"),
    ("argmax", "argument validation", FAMILY_HOST,
     "test_argmax_rejects_before_any_allocation"),
    ("argmax", "Policy-B source temporary allocation", FAMILY_ALLOC,
     "test_argmax_policy_b_temporary_allocation_failure"),
    ("argmax", "int64 destination allocation", FAMILY_ALLOC,
     "test_argmax_destination_allocation_failure_closes_the_temporary"),
    ("argmax", "Policy-B materialization kernel", FAMILY_TRANSFER,
     "test_argmax_materialization_failure_leaks_nothing"),
    ("argmax", "tf_core_argmax", FAMILY_KERNEL,
     "test_argmax_kernel_failure_closes_everything"),
    ("argmax", "wrapper publication", FAMILY_KERNEL,
     "test_argmax_publication_failure_closes_the_core"),
    ("index_select", "argument validation and bounds scan", FAMILY_HOST,
     "test_index_select_rejects_before_any_allocation"),
    ("index_select", "Policy-B source temporary allocation", FAMILY_ALLOC,
     "test_index_select_source_temporary_allocation_failure"),
    ("index_select", "Policy-B index temporary allocation", FAMILY_ALLOC,
     "test_index_select_index_temporary_allocation_failure"),
    ("index_select", "destination allocation", FAMILY_ALLOC,
     "test_index_select_destination_allocation_failure"),
    # Two rows, because this operation reaches ``tf_core_contiguous_copy``
    # at two **different** call sites — ``self._contiguous_temp`` and
    # ``indices._contiguous_temp`` — and an injection that fails the export
    # immediately can only ever reach the first. The second is entered with
    # the source temporary already materialized, and only a journal that
    # delegates call 1 to the real export can get there.
    ("index_select", "Policy-B source materialization kernel",
     FAMILY_TRANSFER,
     "test_index_select_source_materialization_kernel_failure"),
    ("index_select", "Policy-B index materialization kernel",
     FAMILY_TRANSFER,
     "test_index_select_index_materialization_kernel_failure"),
    ("index_select", "tf_core_index_select", FAMILY_KERNEL,
     "test_index_select_kernel_failure_closes_everything"),
    ("index_select", "wrapper publication", FAMILY_KERNEL,
     "test_index_select_publication_failure_closes_the_core"),
    ("index_select", "output count representability", FAMILY_HOST,
     "test_index_select_rejects_an_unrepresentable_output_count"),
)

# Deliberately **not** a matrix row: the reverse-order, exactly-once
# cleanup proof
# (``test_a_failed_index_select_closes_each_allocation_exactly_once``) is
# a cleanup *invariant* checked at an injection position the matrix
# already names — ``tf_core_index_select`` — not a physical seam of its
# own. Listing it would inflate the count by re-describing one position
# as two, so it is traced as a separate cleanup-invariant claim instead.


def test_the_injection_matrix_is_complete_and_every_owner_exists():
    """Traceability, checked rather than promised: every row names either a
    test that exists in this module or an ``N/A`` with a reason, all four
    families appear, and no row is a duplicate."""
    module = sys.modules[__name__]
    seen = set()
    families = set()
    owners = 0
    for operation, position, family, owner in INJECTION_MATRIX:
        key = (operation, position)
        assert key not in seen, key
        seen.add(key)
        families.add(family)
        if owner.startswith("N/A:"):
            assert len(owner) > 40, key   # a reason, not a label
            continue
        owners += 1
        assert hasattr(module, owner), (key, owner)
        assert callable(getattr(module, owner)), owner
    assert families == {FAMILY_HOST, FAMILY_ALLOC, FAMILY_TRANSFER,
                        FAMILY_KERNEL}
    assert owners >= 18, owners
    # ...and every one of the four attacked operations is represented.
    assert {row[0] for row in INJECTION_MATRIX} == {
        "from_int64_array", "int64 contiguous_copy", "argmax", "index_select"}
    # The two Policy-B materialization sites are separate positions with
    # separate owners: index_select materializes through one export twice,
    # and one representative failure may not stand in for both.
    materialization = [row for row in INJECTION_MATRIX
                       if row[0] == "index_select"
                       and "materialization" in row[1]]
    assert len(materialization) == 2, materialization
    assert len({row[3] for row in materialization}) == 2, materialization
    # ...and the exactly-once cleanup proof is deliberately *not* a row: it
    # is an invariant checked at a position already listed, not a seam.
    assert all("exactly once" not in row[1] for row in INJECTION_MATRIX)
    assert all(
        row[3] != "test_a_failed_index_select_closes_each_allocation_"
                  "exactly_once" for row in INJECTION_MATRIX)
    assert hasattr(
        module, "test_a_failed_index_select_closes_each_allocation_"
                "exactly_once")


# ===========================================================================
# 4. Pre-allocation rejection matrices — multi-fault, world-checked
# ===========================================================================

def _stable_object():
    return tensorforge.Tensor([[1.0, 2.0]])


class _ArraySubclass(np.ndarray):
    """An ``ndarray`` subclass, refused by §8.2 step 1."""


FROM_INT64_ARRAY_REJECTIONS = (
    # (label, kwargs, expected exception)
    ("requires_grad wrong type plus malformed array",
     dict(values=[1, 2], requires_grad="yes"), TypeError),
    ("requires_grad True plus malformed array",
     dict(values=[1, 2], requires_grad=True), ValueError),
    ("requires_grad True plus a wrong dtype array",
     dict(values=np.array([1.0, 2.0]), requires_grad=True), ValueError),
    ("a python list",
     dict(values=[1, 2, 3]), TypeError),
    ("a tuple",
     dict(values=(1, 2, 3)), TypeError),
    ("a python int",
     dict(values=7), TypeError),
    ("an ndarray subclass",
     dict(values=np.array([1, 2], dtype=np.int64).view(_ArraySubclass)),
     TypeError),
    ("an int32 array",
     dict(values=np.array([1, 2], dtype=np.int32)), TypeError),
    ("a uint64 array",
     dict(values=np.array([1, 2], dtype=np.uint64)), TypeError),
    ("a bool array",
     dict(values=np.array([True, False])), TypeError),
    ("a float64 array holding integral values",
     dict(values=np.array([1.0, 2.0])), TypeError),
    ("an object array",
     dict(values=np.array([1, 2], dtype=object)), TypeError),
    ("a byte-swapped int64 array",
     dict(values=np.array([1, 2], dtype=">i8")), TypeError),
    ("an empty array",
     dict(values=np.zeros(0, dtype=np.int64)), ValueError),
    ("an empty higher-rank array",
     dict(values=np.zeros((2, 0), dtype=np.int64)), ValueError),
    ("a stable framework object",
     dict(values=_stable_object()), TypeError),
)


@needs_native
@pytest.mark.parametrize("label, kwargs, error",
                         FROM_INT64_ARRAY_REJECTIONS,
                         ids=[row[0] for row in FROM_INT64_ARRAY_REJECTIONS])
def test_from_int64_array_rejections_allocate_nothing_and_move_nothing(
        label, kwargs, error, live_storages, tmp_path):
    """§26.1 in full, driven with calls that are invalid in two ways at
    once where the order is contractual.

    Every rejection is followed by the complete world fingerprint and by a
    proof that the allocator was never entered — which is a different claim
    from "whatever was allocated was released"."""
    sentinels = Sentinels()
    try:
        with unchanged_world(sentinels=sentinels, directory=tmp_path,
                             live=live_storages):
            with no_native_allocation():
                with pytest.raises(error):
                    NativeTensor.from_int64_array(**kwargs)
    finally:
        sentinels.close()


@needs_native
def test_from_int64_array_reports_requires_grad_before_the_array():
    """The precedence §26.2 fixes: ``requires_grad`` is rejected *before*
    the array is examined, so a call malformed in both ways reports the
    keyword rather than the array."""
    with no_native_allocation():
        with pytest.raises(TypeError, match="requires_grad"):
            NativeTensor.from_int64_array([1, 2], requires_grad="yes")
        with pytest.raises(ValueError) as caught:
            NativeTensor.from_int64_array("not an array", requires_grad=True)
    assert "requires_grad" in str(caught.value)


@needs_native
def test_from_int64_array_host_validation_allocates_nothing(live_storages,
                                                            tmp_path):
    """§27.2 family 1, as an **injection** rather than only as a real
    malformed input: the host validator is made to fail and the call
    allocates nothing, publishes nothing, and moves nothing in the
    observable world.

    The ``no_native_allocation`` proof is kept alongside the fingerprint
    rather than replaced by it: "nothing was allocated" and "whatever was
    allocated was released" are different claims, and only the first is
    what this position promises.
    """
    fired = []
    original = cpp._exact_host_array
    sentinels = Sentinels()

    def failing(values, dtype, where):
        fired.append((dtype, where))
        raise Boom("injected: host validation")

    try:
        baseline = settled(live_storages)
        cpp._exact_host_array = failing
        try:
            with unchanged_world(sentinels=sentinels, directory=tmp_path,
                                 live=live_storages):
                with no_native_allocation():
                    with pytest.raises(Boom, match="injected"):
                        NativeTensor.from_int64_array(
                            np.array([1, 2], dtype=np.int64))
        finally:
            cpp._exact_host_array = original
        # Non-vacuity: the seam ran exactly once, at the Core layer, for
        # int64.
        assert fired == [("int64", "NativeTensorCore._from_int64_array")], \
            fired
        assert settled(live_storages) == baseline
        recovered = index(np.array([1, 2], dtype=np.int64))
        try:
            assert exact_ints(recovered.to_numpy()) == [1, 2]
        finally:
            recovered.close()
        assert settled(live_storages) == baseline
    finally:
        sentinels.close()


@needs_native
def test_from_int64_array_asks_the_index_registry_before_the_array(
        live_storages, tmp_path):
    """§26.1 step 2a: the registry gate is asked **before** the input is
    inspected, so an emptied ``INDEX_DTYPES`` closes the door with a
    ``ValueError`` even for a call whose argument would otherwise raise
    ``TypeError`` at the host validator.

    The registry mutation is this test's own deliberate setup, so the
    fingerprint is taken **after** it: what must be proved unchanged is
    what the *rejected call* touches, and a snapshot straddling the
    instrument would confuse the two. The emptied registry is restored in
    a ``finally`` and the ordinary path is exercised afterwards.
    """
    original_index_dtypes = cpp.INDEX_DTYPES
    sentinels = Sentinels()
    try:
        baseline = settled(live_storages)
        cpp.INDEX_DTYPES = ()
        try:
            # Snapshot taken with the instrument already in place, and
            # compared while it is still in place.
            with unchanged_world(sentinels=sentinels, directory=tmp_path,
                                 live=live_storages):
                with no_native_allocation():
                    with pytest.raises(ValueError, match="index dtype"):
                        NativeTensor.from_int64_array("not an array at all")
                    with pytest.raises(ValueError, match="index dtype"):
                        NativeTensor.from_int64_array(
                            np.array([1], dtype=np.int64))
        finally:
            cpp.INDEX_DTYPES = original_index_dtypes
        assert cpp.INDEX_DTYPES == ("int64",)
        assert settled(live_storages) == baseline
        # ...and with the registry restored, the same calls behave normally:
        # the door is open again and the *array* is what is judged.
        with pytest.raises(TypeError):
            NativeTensor.from_int64_array("not an array at all")
        restored = index(np.array([1], dtype=np.int64))
        try:
            assert exact_ints(restored.to_numpy()) == [1]
        finally:
            restored.close()
        assert settled(live_storages) == baseline
    finally:
        cpp.INDEX_DTYPES = original_index_dtypes
        sentinels.close()


ARGMAX_REJECTIONS = (
    ("closed source beats every other fault",
     dict(closed=True, axis="not an int", keepdims="no"), RuntimeError),
    ("an integer source is a role error",
     dict(integer=True, axis=0, keepdims=False), ValueError),
    ("an integer source beats a malformed axis",
     dict(integer=True, axis="not an int", keepdims=False), ValueError),
    ("a bool axis is a type error",
     dict(axis=True, keepdims=False), TypeError),
    ("a float axis is a type error",
     dict(axis=1.0, keepdims=False), TypeError),
    ("a string axis is a type error",
     dict(axis="0", keepdims=False), TypeError),
    ("the axis is validated before keepdims",
     dict(axis="0", keepdims="no"), TypeError),
    ("an out-of-range axis is a value error",
     dict(axis=5, keepdims=False), ValueError),
    ("an out-of-range axis beats a malformed keepdims",
     dict(axis=5, keepdims="no"), ValueError),
    ("a malformed keepdims alone is a type error",
     dict(axis=0, keepdims="no"), TypeError),
)


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("label, spec, error", ARGMAX_REJECTIONS,
                         ids=[row[0] for row in ARGMAX_REJECTIONS])
def test_argmax_rejects_before_any_allocation(label, spec, error, dtype,
                                              live_storages, tmp_path):
    """§17.6's order, driven with multi-fault calls: which of two
    simultaneous defects arrives is the contract, and every rejection
    allocates nothing and moves nothing."""
    spec = dict(spec)
    closed = spec.pop("closed", False)
    integer = spec.pop("integer", False)
    if integer:
        source = index(np.array([[1, 5], [3, 2]], dtype=np.int64))
    else:
        source = floating([[1.0, 5.0], [3.0, 2.0]], dtype)
    sentinels = Sentinels()
    try:
        if closed:
            source.close()
        operands = () if closed else (source,)
        with unchanged_world(operands=operands, sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with no_native_allocation():
                with pytest.raises(error):
                    source.argmax(**spec)
    finally:
        sentinels.close()
        source.close()


@needs_native
def test_argmax_reports_a_non_contiguous_source_the_same_way(live_storages):
    """A non-contiguous source takes the Policy-B path, so its rejections
    must arrive at the same step and still allocate nothing — the
    materialization must not happen before validation."""
    base = floating([[1.0, 5.0], [3.0, 2.0]], "float64")
    view = base.T
    try:
        with no_native_allocation():
            with pytest.raises(TypeError):
                view.argmax(axis=True)
            with pytest.raises(ValueError):
                view.argmax(axis=7)
            with pytest.raises(TypeError):
                view.argmax(axis=0, keepdims="no")
        result = view.argmax(axis=1)
        try:
            assert exact_ints(result.to_numpy()) == [1, 0]
        finally:
            result.close()
    finally:
        view.close()
        base.close()


INDEX_SELECT_REJECTIONS = (
    ("a bool axis beats a non-tensor index",
     dict(axis=True, indices="not a tensor"), TypeError),
    ("a float axis beats a closed source",
     dict(axis=0.0, close_source=True), TypeError),
    ("a non-tensor index beats a closed source",
     dict(axis=0, indices=[0, 1], close_source=True), TypeError),
    ("a numpy index array is not a tensor",
     dict(axis=0, indices=np.array([0, 1], dtype=np.int64)), TypeError),
    ("a python int is not an index tensor",
     dict(axis=0, indices=1), TypeError),
    ("a closed source beats a closed index",
     dict(axis=0, close_source=True, close_index=True), RuntimeError),
    ("a closed index is reported after an open source",
     dict(axis=0, close_index=True), RuntimeError),
    ("an integer source is a role error",
     dict(axis=0, integer_source=True), ValueError),
    ("a floating index is a role error",
     dict(axis=0, floating_index=True), ValueError),
    ("an integer source beats a floating index",
     dict(axis=0, integer_source=True, floating_index=True), ValueError),
    ("a gradient-tracking source names detach",
     dict(axis=0, requires_grad=True), ValueError),
    ("a gradient-tracking source is reported after the dtypes",
     dict(axis=0, requires_grad=True, floating_index=True), ValueError),
    ("an out-of-range axis beats a rank-2 index",
     dict(axis=9, rank2_index=True), ValueError),
    ("a rank-2 index is rejected",
     dict(axis=0, rank2_index=True), ValueError),
    ("a rank-0 index is rejected",
     dict(axis=0, rank0_index=True), ValueError),
    ("a negative index value is rejected, never wrapped",
     dict(axis=0, index_values=[0, -1]), ValueError),
    ("an out-of-range index value is rejected",
     dict(axis=0, index_values=[0, 99]), ValueError),
    ("a rank-2 index beats an out-of-range value",
     dict(axis=0, rank2_index=True, index_values=[0, 99]), ValueError),
)


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("label, spec, error", INDEX_SELECT_REJECTIONS,
                         ids=[row[0] for row in INDEX_SELECT_REJECTIONS])
def test_index_select_rejects_before_any_allocation(label, spec, error,
                                                    dtype, live_storages,
                                                    tmp_path):
    """§18.6's thirteen-step order, driven with calls invalid in two ways
    at once. Every rejection allocates nothing — including the ones that
    happen after the complete bounds scan, which is §13.11's whole point —
    and leaves the observable world byte-identical."""
    spec = dict(spec)
    if spec.pop("integer_source", False):
        source = index(np.array([[1, 5], [3, 2]], dtype=np.int64))
    else:
        source = floating([[1.0, 5.0], [3.0, 2.0]], dtype)
    if spec.pop("requires_grad", False):
        source.close()
        source = floating([[1.0, 5.0], [3.0, 2.0]], dtype)
        source._init_requires_grad(True)
    values = spec.pop("index_values", [0, 1])
    if spec.pop("floating_index", False):
        indices = floating(values, dtype)
    elif spec.pop("rank2_index", False):
        indices = index(np.array([[0], [1]], dtype=np.int64))
    elif spec.pop("rank0_index", False):
        indices = index(np.array(0, dtype=np.int64))
    else:
        indices = index(np.asarray(values, dtype=np.int64))
    if "indices" not in spec:
        spec["indices"] = indices
    close_source = spec.pop("close_source", False)
    close_index = spec.pop("close_index", False)
    sentinels = Sentinels()
    try:
        if close_source:
            source.close()
        if close_index:
            indices.close()
        operands = tuple(
            operand for operand, closed in ((source, close_source),
                                            (indices, close_index))
            if not closed)
        with unchanged_world(operands=operands, sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with no_native_allocation():
                with pytest.raises(error):
                    source.index_select(**spec)
    finally:
        sentinels.close()
        indices.close()
        source.close()


@needs_native
def test_index_select_names_the_offending_index_and_its_position():
    """§14.4: the report names the value and the position, so a caller can
    act on it. Fragments only — the complete wording is not frozen."""
    source = floating([[1.0, 2.0], [3.0, 4.0]], "float64")
    indices = index(np.array([0, 1, 5], dtype=np.int64))
    try:
        with no_native_allocation():
            with pytest.raises(ValueError) as caught:
                source.index_select(0, indices)
        message = str(caught.value)
        assert "5" in message and "2" in message
        assert "position" in message
    finally:
        indices.close()
        source.close()


@needs_native
def test_index_select_rejects_an_unrepresentable_output_count(live_storages,
                                                               tmp_path):
    """§18.6 step 11, the one check this operation needs that no other does:
    ``index_count`` may repeat a position arbitrarily often, so the output
    can be **larger** than the source and its element count is asked about
    on its own, before anything is allocated.

    The real ceiling is unreachable without allocating an impossible
    tensor, so the *instrument* is a temporarily lowered ``_INT64_MAX``.
    The ceiling is lowered first, the fingerprint is taken **after** that
    deliberate setup, and the control at the end proves the lowered ceiling
    really is what rejected the call rather than something else about it.
    """
    original_ceiling = cpp._INT64_MAX
    baseline = settled(live_storages)
    source = floating([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "float64")
    indices = index(np.array([0, 1, 0, 1], dtype=np.int64))
    sentinels = Sentinels()
    here = settled(live_storages)
    try:
        cpp._INT64_MAX = 3
        try:
            with unchanged_world(operands=(source, indices),
                                 sentinels=sentinels, directory=tmp_path,
                                 live=live_storages):
                with no_native_allocation():
                    with pytest.raises(ValueError, match="exceeds"):
                        source.index_select(0, indices)
        finally:
            cpp._INT64_MAX = original_ceiling
        assert cpp._INT64_MAX == original_ceiling
        assert settled(live_storages) == here
        # The instrument's control: the identical call succeeds once the
        # ceiling is back, so the rejection was the ceiling and nothing else.
        result = source.index_select(0, indices)
        try:
            assert result.shape == (4, 3)
        finally:
            result.close()
    finally:
        cpp._INT64_MAX = original_ceiling
        sentinels.close()
        indices.close()
        source.close()
    assert settled(live_storages) == baseline


@needs_native
def test_index_select_rejects_a_gradient_tracking_source_naming_detach():
    """§18.9: rejected rather than silently detached, and the message says
    what to do instead."""
    source = floating([[1.0, 2.0], [3.0, 4.0]], "float64")
    source._init_requires_grad(True)
    indices = index(np.array([1, 0], dtype=np.int64))
    try:
        with no_native_allocation():
            with pytest.raises(ValueError, match="detach"):
                source.index_select(0, indices)
        # ...and the detached route works, giving a plain leaf.
        detached = source.detach()
        try:
            result = detached.index_select(0, indices)
            try:
                assert result.requires_grad is False
                assert result.is_leaf is True
                assert result.grad is None
            finally:
                result.close()
        finally:
            detached.close()
    finally:
        indices.close()
        source.close()


# ===========================================================================
# 5. from_int64_array — every actual allocating step, separately
# ===========================================================================

@needs_fault_injection
def test_from_int64_array_storage_allocation_failure_leaks_nothing(
        live_storages, tmp_path):
    """§27.2 family 2, through the **backend's own** thread-local arm: the
    `int64` storage creation fails inside ``tf_storage_create_typed``, so
    the failure is a real native ``std::bad_alloc`` mapped to
    ``MemoryError`` rather than a Python stand-in."""
    sentinels = Sentinels()
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    before = values.tobytes()
    try:
        with unchanged_world(sentinels=sentinels, directory=tmp_path,
                             live=live_storages):
            with arm_native_allocation(
                    lambda size, dtype: dtype == INDEX_DTYPE) as armed:
                with pytest.raises(MemoryError):
                    NativeTensor.from_int64_array(values)
        assert armed == [(6, INDEX_DTYPE)], armed
        assert values.tobytes() == before, "the host input was mutated"
        recovered = NativeTensor.from_int64_array(values)
        try:
            assert exact_ints(recovered.to_numpy().reshape(-1)) == \
                [1, 2, 3, 4, 5, 6]
        finally:
            recovered.close()
    finally:
        sentinels.close()


@needs_native
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_from_int64_array_transfer_failure_closes_the_storage(
        failure, live_storages, tmp_path):
    """§27.2 family 3: the storage exists and the copy into it fails.

    The storage is **retained in an external list** before the injection
    raises, so its closure is proved by production cleanup while a strong
    reference still holds it — never by ``__del__`` timing."""
    retained = []
    original = cpp.NativeStorage.copy_from
    values = np.array([9, 8, 7], dtype=np.int64)
    before = values.tobytes()
    sentinels = Sentinels()
    try:
        def failing(self, host_values):
            retained.append(self)
            raise failure("injected: host-to-native transfer")

        with unchanged_world(sentinels=sentinels, directory=tmp_path,
                             live=live_storages):
            cpp.NativeStorage.copy_from = failing
            try:
                with pytest.raises(failure, match="injected"):
                    NativeTensor.from_int64_array(values)
            finally:
                cpp.NativeStorage.copy_from = original
        # Non-vacuity: exactly one transfer was attempted, on an int64
        # storage sized from the host array.
        assert len(retained) == 1, retained
        storage = retained[0]
        assert storage.dtype == INDEX_DTYPE and storage.size == 3
        # ...and production cleanup closed it while this reference lives.
        assert storage._handle is None, "the storage was not closed"
        assert values.tobytes() == before
    finally:
        sentinels.close()
    recovered = NativeTensor.from_int64_array(values)
    try:
        assert exact_ints(recovered.to_numpy()) == [9, 8, 7]
    finally:
        recovered.close()


@needs_native
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_from_int64_array_core_construction_failure_closes_the_storage(
        failure, live_storages, tmp_path):
    """The view/core construction step: the storage is fully written and
    the wrapper around it fails."""
    retained = []
    original = cpp._contiguous_view
    values = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sentinels = Sentinels()
    try:
        def failing(storage, dims):
            retained.append(storage)
            raise failure("injected: core construction")

        with unchanged_world(sentinels=sentinels, directory=tmp_path,
                             live=live_storages):
            cpp._contiguous_view = failing
            try:
                with pytest.raises(failure, match="injected"):
                    NativeTensor.from_int64_array(values)
            finally:
                cpp._contiguous_view = original
        assert len(retained) == 1, retained
        storage = retained[0]
        assert storage.dtype == INDEX_DTYPE and storage.size == 4
        assert storage._handle is None, "the storage was not closed"
    finally:
        sentinels.close()


@needs_native
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_from_int64_array_publication_failure_closes_the_core(
        failure, live_storages, tmp_path):
    """§8.5's last step: the core exists and the public wrapper fails.

    The **core** is retained externally, so "it was closed" is a fact about
    production cleanup rather than about reference counting."""
    retained = []
    original = NativeTensor._from_core
    values = np.array([5, 6], dtype=np.int64)
    sentinels = Sentinels()
    try:
        def failing(cls, core, owns_core=True):
            retained.append(core)
            raise failure("injected: wrapper publication")

        with unchanged_world(sentinels=sentinels, directory=tmp_path,
                             live=live_storages):
            NativeTensor._from_core = classmethod(failing)
            try:
                with pytest.raises(failure, match="injected"):
                    NativeTensor.from_int64_array(values)
            finally:
                NativeTensor._from_core = original
        assert len(retained) == 1, retained
        core = retained[0]
        assert core.dtype == INDEX_DTYPE and core.shape == (2,)
        assert core._closed is True, "the core was not closed"
        assert core._storage._handle is None, "the storage was not released"
    finally:
        sentinels.close()


@needs_native
def test_from_int64_array_converts_nothing_and_preserves_rank_and_layout(
        live_storages):
    """The exactness half of the ingress contract, re-proved under the
    hardening lens: nothing converts, rank 0 stays rank 0, a non-contiguous
    exact array is normalized in layout only, and the extremes survive."""
    baseline = settled(live_storages)
    # rank 0 stays rank 0
    scalar = NativeTensor.from_int64_array(np.array(-7, dtype=np.int64))
    try:
        assert scalar.shape == () and scalar.ndim == 0
        assert scalar.item() == -7 and isinstance(scalar.item(), int)
        assert scalar.tolist() == -7
    finally:
        scalar.close()
    # a non-contiguous exact array keeps its logical values
    base = np.arange(12, dtype=np.int64).reshape(3, 4)
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    strided = NativeTensor.from_int64_array(view)
    try:
        assert exact_ints(strided.to_numpy().reshape(-1)) == \
            view.reshape(-1).tolist()
        assert strided.shape == view.shape
        assert strided.contiguous is True
    finally:
        strided.close()
    # the representable extremes round-trip exactly
    extremes = np.array([cpp._INT64_MIN, cpp._INT64_MAX, -1, 0, 1],
                        dtype=np.int64)
    held = NativeTensor.from_int64_array(extremes)
    try:
        assert exact_ints(held.to_numpy()) == extremes.tolist()
        assert held.to_numpy().tobytes() == extremes.tobytes()
        # the caller's array is never aliased
        extremes[0] = 123
        assert exact_ints(held.to_numpy())[0] == cpp._INT64_MIN
    finally:
        held.close()
    assert settled(live_storages) == baseline


@needs_native
def test_no_int64_path_takes_the_uninitialized_allocator(live_storages):
    """§27.3: every integer allocation is zeroed, so the H1 audit table
    gains no row. Proved by watching the uninitialized seam during each
    integer path rather than by reading the code."""
    calls = []
    original = cpp.NativeStorage._uninitialized

    def watching(cls, size, dtype=None, device="cpu"):
        calls.append(dtype)
        return original.__func__(cls, size, dtype=dtype, device=device)

    baseline = settled(live_storages)
    cpp.NativeStorage._uninitialized = classmethod(watching)
    try:
        # Every caller-owned native object here gets its own name and its
        # own ``close()``. An index tensor built inline as an argument
        # would still be the caller's to close, and letting it go would
        # make this test depend on ``__del__`` timing for its cleanup.
        held = index(np.array([1, 2, 3, 4], dtype=np.int64))
        try:
            copy = held.contiguous_copy()
            try:
                source = floating([[1.0, 4.0], [3.0, 2.0]], "float64")
                try:
                    result = source.argmax(axis=1)
                    try:
                        selection_indices = index(
                            np.array([1, 0], dtype=np.int64))
                        try:
                            selection = source.index_select(
                                0, selection_indices)
                            try:
                                assert INDEX_DTYPE not in calls, calls
                            finally:
                                selection.close()
                        finally:
                            selection_indices.close()
                    finally:
                        result.close()
                finally:
                    source.close()
            finally:
                copy.close()
        finally:
            held.close()
    finally:
        cpp.NativeStorage._uninitialized = original
    # ...and every one of them is gone before this test returns.
    assert settled(live_storages) == baseline
    # Non-vacuity: the watcher does see the floating uninitialized path.
    cpp.NativeStorage._uninitialized = classmethod(watching)
    try:
        calls.clear()
        probe = floating([[1.0, 2.0]], "float64")
        try:
            doubled = probe.add(probe)
            try:
                assert calls and all(dtype in FLOATING_DTYPES
                                     for dtype in calls), calls
            finally:
                doubled.close()
        finally:
            probe.close()
    finally:
        cpp.NativeStorage._uninitialized = original
    assert settled(live_storages) == baseline


# ===========================================================================
# 6. int64 contiguous_copy — every allocating step
# ===========================================================================

def _int64_source(contiguous):
    """A contiguous ``int64`` tensor, or a genuinely strided view of one."""
    base = index(np.arange(8, dtype=np.int64))
    if contiguous:
        return base, base
    return strided_view(base), base


@needs_native
@pytest.mark.parametrize("contiguous", [True, False],
                         ids=["contiguous", "non-contiguous"])
def test_int64_contiguous_copy_is_exact_and_independent(contiguous,
                                                        live_storages):
    """The success control the failure rows are measured against: the copy
    owns fresh contiguous storage, reproduces the source's values exactly,
    and a borrowing view never closes its parent."""
    baseline = settled(live_storages)
    source, base = _int64_source(contiguous)
    try:
        expected = exact_ints(source.to_numpy())
        copy = source.contiguous_copy()
        try:
            assert exact_ints(copy.to_numpy()) == expected
            assert copy.to_numpy().tobytes() == source.to_numpy().tobytes()
            assert copy.contiguous is True and copy._core.offset == 0
            assert copy.dtype == INDEX_DTYPE
            assert copy._core.storage is not source._core.storage
            source.close()                    # a view closes only itself
            assert base.closed is (source is base)
            assert exact_ints(copy.to_numpy()) == expected
        finally:
            copy.close()
    finally:
        source.close()
        base.close()
    assert settled(live_storages) == baseline


@needs_fault_injection
@pytest.mark.parametrize("contiguous", [True, False],
                         ids=["contiguous", "non-contiguous"])
def test_int64_contiguous_copy_destination_allocation_failure(
        contiguous, live_storages, tmp_path):
    """The destination allocation, through the backend's own arm."""
    source, base = _int64_source(contiguous)
    sentinels = Sentinels()
    try:
        with unchanged_world(operands=(source,), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with arm_native_allocation(
                    lambda size, dtype: dtype == INDEX_DTYPE) as armed:
                with pytest.raises(MemoryError):
                    source.contiguous_copy()
        assert armed and armed[0][1] == INDEX_DTYPE, armed
        # ...and the same call succeeds once the arm is gone.
        copy = source.contiguous_copy()
        try:
            assert exact_ints(copy.to_numpy()) == exact_ints(
                source.to_numpy())
        finally:
            copy.close()
    finally:
        sentinels.close()
        source.close()
        base.close()


@needs_native
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_int64_contiguous_copy_kernel_failure_closes_the_destination(
        failure, live_storages, tmp_path):
    """The native gather fails after the destination exists. The
    destination core is retained externally so its closure is proved while
    a strong reference holds it."""
    library = cpp._require_library()
    original_kernel = library.tf_core_contiguous_copy
    original_typed = cpp.NativeTensorCore._typed
    retained = []
    fired = []
    source, base = _int64_source(True)
    sentinels = Sentinels()

    def recording_typed(cls, shape, dtype, device="cpu", **kwargs):
        built = original_typed.__func__(cls, shape, dtype, device=device,
                                        **kwargs)
        retained.append(built)
        return built

    def failing_kernel(*arguments):
        fired.append(arguments)
        raise failure("injected: contiguous copy kernel")

    try:
        with unchanged_world(operands=(source,), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            cpp.NativeTensorCore._typed = classmethod(recording_typed)
            library.tf_core_contiguous_copy = failing_kernel
            try:
                with pytest.raises(failure, match="injected"):
                    source.contiguous_copy()
            finally:
                library.tf_core_contiguous_copy = original_kernel
                cpp.NativeTensorCore._typed = original_typed
        assert len(fired) == 1, fired
        assert len(retained) == 1, retained
        destination = retained[0]
        assert destination.dtype == INDEX_DTYPE
        assert destination._closed is True, "the destination was not closed"
        assert destination._storage._handle is None
    finally:
        library.tf_core_contiguous_copy = original_kernel
        cpp.NativeTensorCore._typed = original_typed
        sentinels.close()
        source.close()
        base.close()
    survivor = index(np.array([1, 2], dtype=np.int64))
    try:
        copy = survivor.contiguous_copy()
        try:
            assert exact_ints(copy.to_numpy()) == [1, 2]
        finally:
            copy.close()
    finally:
        survivor.close()


@needs_native
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_int64_contiguous_copy_publication_failure_closes_the_core(
        failure, live_storages, tmp_path):
    retained = []
    original = NativeTensor._from_core
    source, base = _int64_source(True)
    sentinels = Sentinels()
    try:
        def failing(cls, core, owns_core=True):
            retained.append(core)
            raise failure("injected: wrapper publication")

        with unchanged_world(operands=(source,), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            NativeTensor._from_core = classmethod(failing)
            try:
                with pytest.raises(failure, match="injected"):
                    source.contiguous_copy()
            finally:
                NativeTensor._from_core = original
        assert len(retained) == 1, retained
        core = retained[0]
        assert core.dtype == INDEX_DTYPE
        assert core._closed is True and core._storage._handle is None
    finally:
        NativeTensor._from_core = original
        sentinels.close()
        source.close()
        base.close()


@needs_native
def test_closing_an_int64_tensor_is_idempotent_and_rejects_afterwards(
        live_storages):
    """§28, adversarially: close twice, then prove every later operation
    rejects before it validates anything else about the call."""
    baseline = settled(live_storages)
    held = index(np.array([1, 2, 3, 4], dtype=np.int64))
    held.close()
    held.close()                                   # idempotent
    assert held.closed is True
    assert settled(live_storages) == baseline
    with no_native_allocation():
        for call in (lambda: held.contiguous_copy(),
                     lambda: held.to_numpy(),
                     lambda: held.tolist(),
                     lambda: held.reshape((2, 2))):
            with pytest.raises(RuntimeError):
                call()
    # ...and a closed source is reported before a malformed argument.
    source = floating([[1.0, 2.0]], "float64")
    source.close()
    with no_native_allocation():
        with pytest.raises(RuntimeError):
            source.argmax(axis="not an int")
    assert settled(live_storages) == baseline


@needs_native
def test_an_int64_tensor_works_as_a_context_manager(live_storages):
    baseline = settled(live_storages)
    with NativeTensor.from_int64_array(np.array([3, 1], dtype=np.int64)) \
            as held:
        assert exact_ints(held.to_numpy()) == [3, 1]
    assert held.closed is True
    assert settled(live_storages) == baseline


# ===========================================================================
# 7. argmax — every allocating step, at both floating dtypes
# ===========================================================================

def _argmax_case(dtype, contiguous):
    """``(source, base, axis, expected)`` for a contiguous or Policy-B
    argmax whose answer is known independently."""
    base = floating([[1.0, 4.0], [3.0, 2.0]], dtype)
    if contiguous:
        return base, base, 1, [1, 0]
    # The transpose is [[1, 3], [4, 2]]; argmax along axis 1 is [1, 0].
    return base.T, base, 1, [1, 0]


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("contiguous", [True, False],
                         ids=["contiguous", "non-contiguous"])
def test_argmax_succeeds_identically_on_both_layouts(dtype, contiguous,
                                                     live_storages):
    """The success control: the two layouts agree exactly, the result is a
    fresh owning int64 leaf, and it survives the source's close."""
    baseline = settled(live_storages)
    source, base, axis, expected = _argmax_case(dtype, contiguous)
    try:
        result = source.argmax(axis=axis)
        try:
            assert exact_ints(result.to_numpy()) == expected
            assert result.dtype == INDEX_DTYPE
            assert result.requires_grad is False and result.is_leaf is True
            assert result._parents == () and result._op == ""
            source.close()
            base.close()
            assert exact_ints(result.to_numpy()) == expected
        finally:
            result.close()
    finally:
        source.close()
        base.close()
    assert settled(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_argmax_leaves_a_gradient_tracking_source_untouched(dtype,
                                                            live_storages):
    """§17.9: a plain leaf even from a gradient-tracking input, and the
    input's own graph, gradient, and storage are untouched."""
    baseline = settled(live_storages)
    source = floating([[1.0, 4.0], [3.0, 2.0]], dtype)
    source._init_requires_grad(True)
    try:
        before = tensor_view(source)
        result = source.argmax(axis=1)
        try:
            assert result.requires_grad is False
            assert result.is_leaf is True
            assert result.grad is None
            assert exact_ints(result.to_numpy()) == [1, 0]
        finally:
            result.close()
        assert tensor_view(source) == before
        # ...and the source is still usable for a real gradient.
        total = source.sum()
        total.backward()
        assert source.grad is not None
        total.close()
        source.grad.close()
    finally:
        source.close()
    assert settled(live_storages) == baseline


@needs_fault_injection
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_argmax_policy_b_temporary_allocation_failure(dtype, live_storages,
                                                      tmp_path):
    """Position 1 of the non-contiguous path: the private contiguous copy.
    Nothing else has been allocated when it fires."""
    source, base, axis, _ = _argmax_case(dtype, contiguous=False)
    sentinels = Sentinels()
    try:
        with unchanged_world(operands=(source, base), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with arm_native_allocation(
                    lambda size, dtype_: dtype_ == dtype) as armed:
                with pytest.raises(MemoryError):
                    source.argmax(axis=axis)
        assert armed == [(4, dtype)], armed
        result = source.argmax(axis=axis)
        try:
            assert exact_ints(result.to_numpy()) == [1, 0]
        finally:
            result.close()
    finally:
        sentinels.close()
        source.close()
        base.close()


@needs_fault_injection
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("contiguous", [True, False],
                         ids=["contiguous", "non-contiguous"])
def test_argmax_destination_allocation_failure_closes_the_temporary(
        contiguous, dtype, live_storages, tmp_path):
    """Position 2: the ``int64`` destination. On the non-contiguous path it
    fires **after** the Policy-B temporary exists, which is where the
    temporary's cleanup is proved — the reason these are two rows and not
    one."""
    source, base, axis, _ = _argmax_case(dtype, contiguous)
    sentinels = Sentinels()
    try:
        with unchanged_world(operands=(source, base), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with arm_native_allocation(
                    lambda size, dtype_: dtype_ == INDEX_DTYPE) as armed:
                with pytest.raises(MemoryError):
                    source.argmax(axis=axis)
        assert armed == [(2, INDEX_DTYPE)], armed
        result = source.argmax(axis=axis)
        try:
            assert exact_ints(result.to_numpy()) == [1, 0]
        finally:
            result.close()
    finally:
        sentinels.close()
        source.close()
        base.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_argmax_materialization_failure_leaks_nothing(failure, dtype,
                                                      live_storages,
                                                      tmp_path):
    """The Policy-B materialization **kernel**, distinct from the
    allocation that precedes it: the temporary's storage exists and the
    gather into it fails."""
    library = cpp._require_library()
    original_kernel = library.tf_core_contiguous_copy
    fired = []
    source, base, axis, _ = _argmax_case(dtype, contiguous=False)
    sentinels = Sentinels()

    def failing_kernel(*arguments):
        fired.append(arguments)
        raise failure("injected: materialization")

    try:
        with unchanged_world(operands=(source, base), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            library.tf_core_contiguous_copy = failing_kernel
            try:
                with pytest.raises(failure, match="injected"):
                    source.argmax(axis=axis)
            finally:
                library.tf_core_contiguous_copy = original_kernel
        assert len(fired) == 1, fired
        result = source.argmax(axis=axis)
        try:
            assert exact_ints(result.to_numpy()) == [1, 0]
        finally:
            result.close()
    finally:
        library.tf_core_contiguous_copy = original_kernel
        sentinels.close()
        source.close()
        base.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
@pytest.mark.parametrize("contiguous", [True, False],
                         ids=["contiguous", "non-contiguous"])
def test_argmax_kernel_failure_closes_everything(contiguous, failure, dtype,
                                                 live_storages, tmp_path):
    """``tf_core_argmax`` fails after every allocation exists. The
    destination is retained externally, so its closure is a fact about
    production cleanup."""
    library = cpp._require_library()
    original_kernel = library.tf_core_argmax
    original_typed = cpp.NativeTensorCore._typed
    retained = []
    fired = []
    source, base, axis, _ = _argmax_case(dtype, contiguous)
    sentinels = Sentinels()

    def recording_typed(cls, shape, dtype_, device="cpu", **kwargs):
        built = original_typed.__func__(cls, shape, dtype_, device=device,
                                        **kwargs)
        if dtype_ == INDEX_DTYPE:
            retained.append(built)
        return built

    def failing_kernel(*arguments):
        fired.append(arguments)
        raise failure("injected: argmax kernel")

    try:
        with unchanged_world(operands=(source, base), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            cpp.NativeTensorCore._typed = classmethod(recording_typed)
            library.tf_core_argmax = failing_kernel
            try:
                with pytest.raises(failure, match="injected"):
                    source.argmax(axis=axis)
            finally:
                library.tf_core_argmax = original_kernel
                cpp.NativeTensorCore._typed = original_typed
        assert len(fired) == 1, fired
        assert len(retained) == 1, retained
        destination = retained[0]
        assert destination.dtype == INDEX_DTYPE
        assert destination._closed is True, "the destination was not closed"
        assert destination._storage._handle is None
        result = source.argmax(axis=axis)
        try:
            assert exact_ints(result.to_numpy()) == [1, 0]
        finally:
            result.close()
    finally:
        library.tf_core_argmax = original_kernel
        cpp.NativeTensorCore._typed = original_typed
        sentinels.close()
        source.close()
        base.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_argmax_publication_failure_closes_the_core(failure, dtype,
                                                    live_storages,
                                                    tmp_path):
    retained = []
    original = NativeTensor._from_core
    source, base, axis, _ = _argmax_case(dtype, contiguous=True)
    sentinels = Sentinels()
    try:
        def failing(cls, core, owns_core=True):
            retained.append(core)
            raise failure("injected: wrapper publication")

        with unchanged_world(operands=(source,), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            NativeTensor._from_core = classmethod(failing)
            try:
                with pytest.raises(failure, match="injected"):
                    source.argmax(axis=axis)
            finally:
                NativeTensor._from_core = original
        assert len(retained) == 1, retained
        core = retained[0]
        assert core.dtype == INDEX_DTYPE
        assert core._closed is True and core._storage._handle is None
    finally:
        NativeTensor._from_core = original
        sentinels.close()
        source.close()
        base.close()


# ===========================================================================
# 8. index_select — every allocating step, at both floating dtypes
# ===========================================================================

def _index_select_case(dtype, source_contiguous, index_contiguous):
    """``(source, source_base, indices, index_base, expected_host)``.

    The shapes are chosen so the three allocations this operation can make
    have **distinct sizes**: the source temporary holds 6 elements, the
    index temporary 4, and the destination 8. That is what lets each be
    armed on its own rather than by an ordinal that could drift.
    """
    source_base = floating([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
    source = source_base if source_contiguous else source_base.T
    index_base = index(np.array([1, 9, 0, 9, 1, 9, 0, 9], dtype=np.int64))
    if index_contiguous:
        index_base.close()
        index_base = index(np.array([1, 0, 1, 0], dtype=np.int64))
        indices = index_base
    else:
        indices = strided_view(index_base)      # [1, 0, 1, 0], 4 elements
    host = source.to_numpy()
    expected = np.take(host, np.array([1, 0, 1, 0], dtype=np.int64), axis=0)
    return source, source_base, indices, index_base, expected


def _close_case(case):
    source, source_base, indices, index_base, _ = case
    indices.close()
    index_base.close()
    source.close()
    source_base.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("source_contiguous, index_contiguous", [
    (True, True), (False, True), (True, False), (False, False),
], ids=["both-contiguous", "strided-source", "strided-index", "both-strided"])
def test_index_select_succeeds_on_every_layout_combination(
        source_contiguous, index_contiguous, dtype, live_storages):
    """The success control for the four layout combinations, compared as
    raw IEEE-754 bits and never as numbers. Duplicates and order are
    preserved exactly."""
    baseline = settled(live_storages)
    case = _index_select_case(dtype, source_contiguous, index_contiguous)
    source, source_base, indices, index_base, expected = case
    try:
        result = source.index_select(0, indices)
        try:
            assert result.dtype == source.dtype
            assert result.shape == expected.shape
            assert same_bits(result.to_numpy(), expected)
            assert result.contiguous is True and result._core.offset == 0
            assert result.requires_grad is False and result.is_leaf is True
            # duplicates give independent copies in their original positions
            values = result.to_numpy()
            assert bits(values[0]) == bits(values[2])
            assert bits(values[1]) == bits(values[3])
        finally:
            result.close()
    finally:
        _close_case(case)
    assert settled(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_index_select_moves_exceptional_values_by_object_representation(
        dtype, live_storages):
    """§18.4: values cross by object representation, so both signed zeros,
    both infinities, subnormals, and NaN payloads survive bit for bit."""
    baseline = settled(live_storages)
    numpy_dtype = NUMPY_DTYPE[dtype]
    tiny = np.finfo(numpy_dtype).tiny
    payload = np.array([[0.0, -0.0, np.inf],
                        [-np.inf, tiny / 4.0, np.nan]], dtype=numpy_dtype)
    source = NativeTensor.from_array(payload.astype(np.float64), dtype=dtype)
    # A source built through the float64 ingress would round; rebuild the
    # exact bits through the storage so the comparison is honest.
    source._core.storage.copy_from(payload.reshape(-1))
    indices = index(np.array([1, 0, 1], dtype=np.int64))
    try:
        expected = np.take(payload, np.array([1, 0, 1], dtype=np.int64),
                           axis=0)
        result = source.index_select(0, indices)
        try:
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        indices.close()
        source.close()
    assert settled(live_storages) == baseline


@needs_fault_injection
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_index_select_source_temporary_allocation_failure(dtype,
                                                          live_storages,
                                                          tmp_path):
    """Position 1: the source's Policy-B temporary, on a call where the
    index operand is contiguous so this is the **only** temporary."""
    case = _index_select_case(dtype, False, True)
    source, source_base, indices, index_base, expected = case
    sentinels = Sentinels()
    try:
        with unchanged_world(operands=(source, indices), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with arm_native_allocation(
                    lambda size, dtype_: dtype_ == dtype and size == 6) \
                    as armed:
                with pytest.raises(MemoryError):
                    source.index_select(0, indices)
        assert armed == [(6, dtype)], armed
        result = source.index_select(0, indices)
        try:
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        sentinels.close()
        _close_case(case)


@needs_fault_injection
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_index_select_index_temporary_allocation_failure(dtype,
                                                         live_storages,
                                                         tmp_path):
    """Position 2: the **index** temporary, on a call where the source
    temporary has already been allocated. This is where reverse-order
    cleanup of the source temporary is proved — the reason this is its own
    row rather than a repeat of position 1."""
    case = _index_select_case(dtype, False, False)
    source, source_base, indices, index_base, expected = case
    sentinels = Sentinels()
    try:
        with unchanged_world(operands=(source, indices), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with arm_native_allocation(
                    lambda size, dtype_: dtype_ == INDEX_DTYPE and size == 4) \
                    as armed:
                with pytest.raises(MemoryError):
                    source.index_select(0, indices)
        assert armed == [(4, INDEX_DTYPE)], armed
        result = source.index_select(0, indices)
        try:
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        sentinels.close()
        _close_case(case)


@needs_fault_injection
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_index_select_destination_allocation_failure(dtype, live_storages,
                                                     tmp_path):
    """Position 3: the destination, on a call where **both** temporaries
    already exist — so this row proves both of them are closed, in reverse
    allocation order."""
    case = _index_select_case(dtype, False, False)
    source, source_base, indices, index_base, expected = case
    sentinels = Sentinels()
    try:
        with unchanged_world(operands=(source, indices), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with arm_native_allocation(
                    lambda size, dtype_: dtype_ == dtype and size == 8) \
                    as armed:
                with pytest.raises(MemoryError):
                    source.index_select(0, indices)
        assert armed == [(8, dtype)], armed
        result = source.index_select(0, indices)
        try:
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        sentinels.close()
        _close_case(case)


@contextlib.contextmanager
def journalled_materialization(fail_on, failure):
    """Instrument ``tf_core_contiguous_copy`` with a call **journal** that
    delegates every call before ``fail_on`` to the real export and fails
    exactly the ``fail_on``-th.

    ``index_select`` reaches this one export at **two** different call
    sites on a both-strided call — ``self._contiguous_temp`` for the
    floating source, then ``indices._contiguous_temp`` for the ``int64``
    index — and an injection that fails immediately can only ever reach the
    first. Delegating call 1 to the real export is what makes call 2 a
    genuinely different physical position, entered with the source
    temporary already fully materialized, rather than the same position
    reached twice.

    Yields ``(journal, retained)``. The journal carries the dtype, element
    count, and rank of each call, which is what tells the two sites apart
    by metadata rather than by order, plus the raw bytes of every
    temporary already completed when that call began. ``retained`` holds
    every Core allocated while the journal was installed, strongly, so
    closure is proved against a live reference instead of ``__del__``.
    """
    library = cpp._require_library()
    original_kernel = library.tf_core_contiguous_copy
    original_typed = cpp.NativeTensorCore._typed
    original_uninit = cpp.NativeTensorCore._uninitialized
    journal = []
    retained = []

    # Both Core allocators, because this path uses both: the floating arm
    # of ``contiguous_copy`` keeps the H1 uninitialized allocation while
    # the index arm and the destination take the zeroed typed one (§27.3).
    def recording_typed(cls, shape, dtype, device="cpu", **kwargs):
        built = original_typed.__func__(cls, shape, dtype, device=device,
                                        **kwargs)
        retained.append(built)
        return built

    def recording_uninit(cls, shape, dtype="float64", device="cpu"):
        built = original_uninit.__func__(cls, shape, dtype, device=device)
        retained.append(built)
        return built

    def journalling_kernel(*arguments):
        # ``contiguous_copy`` allocates its destination before calling the
        # export, so the newest retained Core *is* this call's destination.
        destination = retained[-1]
        journal.append({
            "call": len(journal) + 1,
            "dtype": destination.dtype,
            "numel": destination.numel,
            "rank": int(arguments[5]),
            # Every temporary finished before this call began — empty on
            # call 1, and on call 2 the proof that call 1 really ran.
            "completed": tuple(raw_bytes(core) for core in retained[:-1]),
        })
        if len(journal) == fail_on:
            raise failure("injected: materialization")
        return original_kernel(*arguments)

    cpp.NativeTensorCore._typed = classmethod(recording_typed)
    cpp.NativeTensorCore._uninitialized = classmethod(recording_uninit)
    library.tf_core_contiguous_copy = journalling_kernel
    try:
        yield journal, retained
    finally:
        library.tf_core_contiguous_copy = original_kernel
        cpp.NativeTensorCore._typed = original_typed
        cpp.NativeTensorCore._uninitialized = original_uninit


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_index_select_source_materialization_kernel_failure(failure, dtype,
                                                            live_storages,
                                                            tmp_path):
    """§27.2 family 3 at the **first** of this operation's two Policy-B
    materialization call sites: the floating source copy, which fails
    before the index operand has been looked at."""
    case = _index_select_case(dtype, False, False)
    source, source_base, indices, index_base, expected = case
    sentinels = Sentinels()
    here = settled(live_storages)
    try:
        with unchanged_world(operands=(source, indices), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with journalled_materialization(1, failure) as journalled:
                journal, retained = journalled
                with pytest.raises(failure, match="injected"):
                    source.index_select(0, indices)
        # Exactly one call reached the export, and its metadata is the
        # source's: the floating temporary, 6 elements, rank 2.
        assert len(journal) == 1, journal
        assert journal[0]["dtype"] == dtype
        assert journal[0]["numel"] == 6
        assert journal[0]["rank"] == 2
        assert journal[0]["completed"] == ()
        # The index temporary and the destination were never allocated at
        # all, so this failure position is upstream of both.
        assert [core.dtype for core in retained] == [dtype], retained
        assert retained[0]._closed is True
        assert retained[0]._storage._handle is None
        assert settled(live_storages) == here
        result = source.index_select(0, indices)
        try:
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        sentinels.close()
        _close_case(case)


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_index_select_index_materialization_kernel_failure(failure, dtype,
                                                           live_storages,
                                                           tmp_path):
    """§27.2 family 3 at the **second** call site: the ``int64`` index
    copy, entered only after the floating source temporary was really
    materialized by the real export.

    This is a different physical position from the source materialization
    above, not the same one reached twice, and the journal proves it two
    ways: the two calls differ in dtype, element count, and rank, and call
    2 begins with call 1's destination already holding the source's
    values.
    """
    case = _index_select_case(dtype, False, False)
    source, source_base, indices, index_base, expected = case
    sentinels = Sentinels()
    # What a fully materialized source temporary must contain: this
    # (non-contiguous) source's own values, in logical order, as raw bytes.
    materialized = np.ascontiguousarray(source.to_numpy()).tobytes()
    here = settled(live_storages)
    try:
        with unchanged_world(operands=(source, indices), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            with journalled_materialization(2, failure) as journalled:
                journal, retained = journalled
                with pytest.raises(failure, match="injected"):
                    source.index_select(0, indices)
        assert len(journal) == 2, journal
        first, second = journal
        # Distinguishable by metadata, never by ordering alone.
        assert (first["dtype"], first["numel"], first["rank"]) == (dtype, 6, 2)
        assert (second["dtype"], second["numel"], second["rank"]) == \
            (INDEX_DTYPE, 4, 1)
        assert first["dtype"] != second["dtype"]
        assert first["numel"] != second["numel"]
        assert first["rank"] != second["rank"]
        # Call 1 was delegated to the real export, so the source temporary
        # was genuinely materialized before call 2 was allowed to fail.
        assert first["completed"] == ()
        assert second["completed"] == (materialized,)
        # Both temporaries exist; **no destination was allocated after the
        # second materialization failed**, which is the whole point of
        # rejecting at this position rather than later.
        assert [core.dtype for core in retained] == [dtype, INDEX_DTYPE], \
            [core.dtype for core in retained]
        assert [core.numel for core in retained] == [6, 4]
        # Both are closed while this test still holds them.
        for core in retained:
            assert core._closed is True, core
            assert core._storage._handle is None, core
        assert settled(live_storages) == here
        # The operands themselves are untouched, exactly.
        assert same_bits(source.to_numpy(), np.frombuffer(
            materialized, dtype=NUMPY_DTYPE[dtype]).reshape(source.shape))
        assert exact_ints(indices.to_numpy()) == [1, 0, 1, 0]
        result = source.index_select(0, indices)
        try:
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        sentinels.close()
        _close_case(case)


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_index_select_kernel_failure_closes_everything(failure, dtype,
                                                       live_storages,
                                                       tmp_path):
    """``tf_core_index_select`` fails with both temporaries and the
    destination alive. All three are retained externally and each is proved
    closed exactly once, in reverse allocation order."""
    library = cpp._require_library()
    original_kernel = library.tf_core_index_select
    original_typed = cpp.NativeTensorCore._typed
    original_uninit = cpp.NativeTensorCore._uninitialized
    retained = []
    fired = []
    case = _index_select_case(dtype, False, False)
    source, source_base, indices, index_base, expected = case
    sentinels = Sentinels()

    # **Both** Core allocators are recorded, because this path uses both:
    # ``contiguous_copy``'s floating arm keeps the H1 uninitialized
    # allocation while its index arm and the destination take the zeroed
    # typed one (§27.3). Wrapping only one would silently miss an operand
    # and turn a three-allocation claim into a two-allocation one.
    def recording_typed(cls, shape, dtype_, device="cpu", **kwargs):
        built = original_typed.__func__(cls, shape, dtype_, device=device,
                                        **kwargs)
        retained.append(built)
        return built

    def recording_uninit(cls, shape, dtype="float64", device="cpu"):
        built = original_uninit.__func__(cls, shape, dtype, device=device)
        retained.append(built)
        return built

    def failing_kernel(*arguments):
        fired.append(arguments)
        raise failure("injected: index_select kernel")

    try:
        with unchanged_world(operands=(source, indices), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            cpp.NativeTensorCore._typed = classmethod(recording_typed)
            cpp.NativeTensorCore._uninitialized = classmethod(recording_uninit)
            library.tf_core_index_select = failing_kernel
            try:
                with pytest.raises(failure, match="injected"):
                    source.index_select(0, indices)
            finally:
                library.tf_core_index_select = original_kernel
                cpp.NativeTensorCore._typed = original_typed
                cpp.NativeTensorCore._uninitialized = original_uninit
        assert len(fired) == 1, fired
        # Three allocations, in order: source temporary, index temporary,
        # destination — and every one of them closed.
        assert [built.dtype for built in retained] == \
            [dtype, INDEX_DTYPE, dtype], [b.dtype for b in retained]
        assert [built.numel for built in retained] == [6, 4, 8]
        for built in retained:
            assert built._closed is True, built
            assert built._storage._handle is None, built
        result = source.index_select(0, indices)
        try:
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        library.tf_core_index_select = original_kernel
        cpp.NativeTensorCore._typed = original_typed
        cpp.NativeTensorCore._uninitialized = original_uninit
        sentinels.close()
        _close_case(case)


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("failure", [Boom, Abort],
                         ids=["Exception", "BaseException"])
def test_index_select_publication_failure_closes_the_core(failure, dtype,
                                                          live_storages,
                                                          tmp_path):
    """The last seam on the path, at **both** floating widths independently.

    Each width is proved only against itself — the destination Core carries
    this run's source dtype and nothing is compared across the two — because
    a cross-width numeric claim would be a mixed-precision statement this
    runtime does not make.
    """
    retained = []
    original = NativeTensor._from_core
    case = _index_select_case(dtype, True, True)
    source, source_base, indices, index_base, expected = case
    sentinels = Sentinels()
    here = settled(live_storages)
    try:
        def failing(cls, core, owns_core=True):
            retained.append(core)
            raise failure("injected: wrapper publication")

        with unchanged_world(operands=(source, indices), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            NativeTensor._from_core = classmethod(failing)
            try:
                with pytest.raises(failure, match="injected"):
                    source.index_select(0, indices)
            finally:
                NativeTensor._from_core = original
        # The injection fired exactly once, and on a destination carrying
        # **this run's** dtype.
        assert len(retained) == 1, retained
        core = retained[0]
        assert core.dtype == dtype and core.numel == expected.size
        # Closed, and its handle released, while this test still holds it.
        assert core._closed is True and core._storage._handle is None
        assert settled(live_storages) == here
        result = source.index_select(0, indices)
        try:
            assert result.dtype == dtype
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        NativeTensor._from_core = original
        sentinels.close()
        _close_case(case)


@needs_native
def test_a_failed_index_select_closes_each_allocation_exactly_once(
        live_storages, tmp_path):
    """"Closed" and "closed exactly once" are different claims, and the
    second is the one a double-free would break. Every ``NativeStorage``
    release is counted per object across a failed three-allocation call.

    This is a cleanup **invariant** checked at a position ``INJECTION_MATRIX``
    already names (``tf_core_index_select``), not a physical seam of its
    own — which is why it is deliberately absent from that matrix. It
    carries the complete world fingerprint all the same, so the counted
    failure is held to exactly the standard every other injected failure
    is.
    """
    library = cpp._require_library()
    original_kernel = library.tf_core_index_select
    original_close = cpp.NativeStorage.close
    releases = {}

    def counting_close(self):
        if self._handle is not None:
            releases[id(self)] = releases.get(id(self), 0) + 1
        original_close(self)

    def failing_kernel(*arguments):
        raise Boom("injected: index_select kernel")

    case = _index_select_case("float64", False, False)
    source, source_base, indices, index_base, expected = case
    sentinels = Sentinels()
    here = settled(live_storages)
    try:
        with unchanged_world(operands=(source, indices), sentinels=sentinels,
                             directory=tmp_path, live=live_storages):
            cpp.NativeStorage.close = counting_close
            library.tf_core_index_select = failing_kernel
            try:
                with pytest.raises(Boom, match="injected"):
                    source.index_select(0, indices)
            finally:
                library.tf_core_index_select = original_kernel
                cpp.NativeStorage.close = original_close
        # Three storages were released by the failure, and each exactly once.
        assert len(releases) == 3, releases
        assert set(releases.values()) == {1}, releases
        assert settled(live_storages) == here
        # ...and the operation still works once the injection is gone.
        result = source.index_select(0, indices)
        try:
            assert same_bits(result.to_numpy(), expected)
        finally:
            result.close()
    finally:
        cpp.NativeStorage.close = original_close
        library.tf_core_index_select = original_kernel
        sentinels.close()
        _close_case(case)


@needs_native
def test_an_index_select_result_survives_both_operands_closing(
        live_storages):
    """§28: the delivered output is the caller's, owns fresh storage, and
    outlives everything it was computed from."""
    baseline = settled(live_storages)
    case = _index_select_case("float64", True, True)
    source, source_base, indices, index_base, expected = case
    result = source.index_select(0, indices)
    try:
        _close_case(case)
        assert same_bits(result.to_numpy(), expected)
    finally:
        result.close()
    assert settled(live_storages) == baseline


# ===========================================================================
# 9. The C ABI as an independent authority — argmax
# ===========================================================================
#
# The two exports have **different** validation lists (§22.10), so they get
# different matrices. No blanket helper covers both: one would obscure which
# rule each operation owns, which is the defect the contract names.

# A destination sentinel no computed index could ever equal, so "unchanged"
# is a fact about bytes rather than a coincidence of values.
ARGMAX_SENTINEL = [-424242424242, 987654321987]


def _argmax_abi_fixture():
    """A 2x3 floating source and a 2-element int64 destination prefilled
    with the sentinel."""
    source = typed_core((6,), "float64", [1.0, 7.0, 3.0, 9.0, 9.0, 2.0])
    destination = typed_core((2,), INDEX_DTYPE, ARGMAX_SENTINEL)
    return source, destination


ARGMAX_ABI_CASES = (
    # (label, builder) -> a callable taking (library, source, destination)
    ("null source", lambda lib, s, d: lib.tf_core_argmax(
        None, 0, d._storage._require_open(), 2, 3, 1)),
    ("null destination", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, None, 2, 3, 1)),
    ("zero outer", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), 0, 3, 1)),
    ("negative outer", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), -2, 3, 1)),
    ("zero axis_length", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), 2, 0, 1)),
    ("negative axis_length", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), 2, -3, 1)),
    ("zero inner", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), 2, 3, 0)),
    ("negative inner", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), 2, 3, -1)),
    ("source product overflow", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(),
        (2 ** 63 - 1) // 2, 4, 1)),
    # An overflow reached through ``inner`` rather than through
    # ``axis_length``. It is deliberately **not** labelled as the export's
    # "output index count overflows int64" branch: ``index_count`` is
    # ``outer * inner`` and the source product is ``outer * axis_length *
    # inner`` with ``axis_length >= 1``, so the source product is checked
    # first and always overflows at least as early. That branch is
    # unreachable by construction, and claiming this case reaches it would
    # be exactly the kind of mislabelling §27.2 forbids for injections.
    ("product overflow reached through inner",
     lambda lib, s, d: lib.tf_core_argmax(
         s._storage._require_open(), 0, d._storage._require_open(),
         (2 ** 63 - 1) // 2, 1, 4)),
    ("negative source offset", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), -1, d._storage._require_open(), 2, 3, 1)),
    ("source span too long", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), 2, 4, 1)),
    ("offset pushes the source span out", lambda lib, s, d:
        lib.tf_core_argmax(
            s._storage._require_open(), 2, d._storage._require_open(),
            2, 3, 1)),
    ("destination too small", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), 3, 2, 1)),
    ("destination too large", lambda lib, s, d: lib.tf_core_argmax(
        s._storage._require_open(), 0, d._storage._require_open(), 1, 6, 1)),
)


@needs_native
@pytest.mark.parametrize("label, call", ARGMAX_ABI_CASES,
                         ids=[row[0] for row in ARGMAX_ABI_CASES])
def test_the_argmax_abi_rejects_malformed_metadata_and_writes_nothing(
        label, call, live_storages):
    """§22.8's list, driven directly through ``ctypes`` with the Python
    validators bypassed entirely.

    Each case clears the slot, prefills the destination with a distinctive
    sentinel, captures every operand's bytes, makes the call, requires a
    C-side rejection, and then proves not one byte of the destination or of
    the source moved.
    """
    library = cpp._require_library()
    library.tf_clear_error()
    source, destination = _argmax_abi_fixture()
    baseline = settled(live_storages)
    try:
        source_before = raw_bytes(source)
        destination_before = raw_bytes(destination)
        with no_native_allocation():
            with pytest.raises(ValueError) as caught:
                call(library, source, destination)
        assert "argmax" in str(caught.value)
        assert raw_bytes(destination) == destination_before, label
        assert raw_bytes(source) == source_before, label
        assert settled(live_storages) == baseline
        assert error_slot() == cpp.TF_OK, "the errcheck hook left a stale code"
    finally:
        destination.close()
        source.close()
        library.tf_clear_error()


# (label, source dtype and values, destination dtype and values, message)
ARGMAX_ROLE_REJECTIONS = (
    ("an int64 source is a role error",
     (INDEX_DTYPE, [1, 5, 2, 3]), (INDEX_DTYPE, [-77]), "floating"),
    ("an int64 source with a floating destination is still a source role "
     "error", (INDEX_DTYPE, [1, 5, 2, 3]), ("float64", [-7.5]), "floating"),
    ("a float64 destination is a role error",
     ("float64", [1.0, 5.0, 2.0, 3.0]), ("float64", [-7.5]), "int64"),
    ("a float32 destination is a role error",
     ("float64", [1.0, 5.0, 2.0, 3.0]), ("float32", [-7.5]), "int64"),
    ("a float32 source with a float32 destination is a destination role "
     "error", ("float32", [1.0, 5.0, 2.0, 3.0]), ("float32", [-7.5]),
     "int64"),
    ("a float32 source with a float64 destination is a destination role "
     "error", ("float32", [1.0, 5.0, 2.0, 3.0]), ("float64", [-7.5]),
     "int64"),
)


@needs_native
@pytest.mark.parametrize("label, source_spec, destination_spec, message",
                         ARGMAX_ROLE_REJECTIONS,
                         ids=[row[0] for row in ARGMAX_ROLE_REJECTIONS])
def test_the_argmax_abi_rejects_every_dtype_role_error(
        label, source_spec, destination_spec, message, live_storages):
    """The role half of §22.8, held to the same standard as the malformed
    metadata matrix: **both** operands' bytes are captured and compared, no
    native storage is allocated, and the error slot is clean once Python's
    ``errcheck`` hook has handled the rejection.

    This is where the two exports differ: the source must be floating and
    the destination must be exactly ``int64`` — and **neither**
    ``require_floating`` nor ``require_matching_dtype`` is applied to that
    destination, which the valid control below proves by succeeding.
    """
    library = cpp._require_library()
    library.tf_clear_error()
    baseline = settled(live_storages)
    source = typed_core((4,), source_spec[0], source_spec[1])
    destination = typed_core((1,), destination_spec[0], destination_spec[1])
    try:
        source_before = raw_bytes(source)
        destination_before = raw_bytes(destination)
        with no_native_allocation():
            with pytest.raises(ValueError, match=message):
                library.tf_core_argmax(
                    source._storage._require_open(), 0,
                    destination._storage._require_open(), 1, 4, 1)
        assert raw_bytes(source) == source_before, label
        assert raw_bytes(destination) == destination_before, label
        assert settled(live_storages) == baseline + 2, label
        assert error_slot() == cpp.TF_OK, "the errcheck hook left a stale code"
    finally:
        destination.close()
        source.close()
    assert settled(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_argmax_abi_accepts_the_one_valid_mixed_role_call(dtype,
                                                              live_storages):
    """The control the rejections above are measured against: the valid
    mixed-role call — floating source, ``int64`` destination — which
    *either* forbidden guard would have refused, succeeds at both source
    widths and writes the answer."""
    library = cpp._require_library()
    library.tf_clear_error()
    baseline = settled(live_storages)
    valid_source = typed_core((6,), dtype, [1.0, 7.0, 3.0, 9.0, 9.0, 2.0])
    valid_destination = typed_core((2,), INDEX_DTYPE, ARGMAX_SENTINEL)
    try:
        library.tf_core_argmax(
            valid_source._storage._require_open(), 0,
            valid_destination._storage._require_open(), 2, 3, 1)
        assert exact_ints(valid_destination.to_numpy()) == [1, 0]
        assert valid_source.dtype != valid_destination.dtype
        assert error_slot() == cpp.TF_OK
    finally:
        valid_destination.close()
        valid_source.close()
    assert settled(live_storages) == baseline


@needs_native
def test_the_argmax_abi_rejects_a_self_aliasing_handle(live_storages):
    """Aliasing is defense in depth here — the dtype roles already make a
    genuine alias unreachable — so what is observable, and asserted, is
    that the same handle in both positions is rejected and the storage is
    left byte-for-byte unchanged, at every dtype."""
    library = cpp._require_library()
    baseline = settled(live_storages)
    for dtype, values in (("float64", [1.0, 2.0]), ("float32", [1.0, 2.0]),
                          (INDEX_DTYPE, [11, 22])):
        storage = typed_core((2,), dtype, values)
        before = raw_bytes(storage)
        try:
            with pytest.raises(ValueError):
                library.tf_core_argmax(
                    storage._storage._require_open(), 0,
                    storage._storage._require_open(), 2, 1, 1)
            assert raw_bytes(storage) == before, dtype
        finally:
            storage.close()
    assert settled(live_storages) == baseline


@needs_native
def test_the_argmax_destination_sentinel_detects_a_single_element_write():
    """The non-vacuity control for every "not one byte moved" assertion
    above: the same comparison must notice one written index."""
    library = cpp._require_library()
    source = typed_core((4,), "float64", [1.0, 4.0, 2.0, 3.0])
    destination = typed_core((2,), INDEX_DTYPE, ARGMAX_SENTINEL)
    try:
        before = raw_bytes(destination)
        # A valid call writing exactly one of the two indices is impossible
        # through this export — its destination capacity is exact — so the
        # single-element sensitivity is proved on the host instead, and the
        # export's own write is proved separately.
        mutated = np.frombuffer(before, dtype=np.int64).copy()
        mutated[1] += 1
        assert mutated.tobytes() != before, "the byte comparison is blind"
        library.tf_core_argmax(
            source._storage._require_open(), 0,
            destination._storage._require_open(), 2, 2, 1)
        assert raw_bytes(destination) != before
        assert exact_ints(destination.to_numpy()) == [1, 1]
    finally:
        destination.close()
        source.close()


# ===========================================================================
# 10. The C ABI as an independent authority — index_select
# ===========================================================================

FLOAT_SENTINEL = [-1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5,
                  -9.5, -10.5, -11.5, -12.5, -13.5, -14.5, -15.5, -16.5]


def _index_select_abi_fixture(dtype="float64", index_values=(1, 0, 1),
                              destination_size=None):
    """A 2x3 source, an int64 index vector, and a floating destination
    prefilled with a distinctive sentinel."""
    source = typed_core((6,), dtype, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    indices = typed_core((len(index_values),), INDEX_DTYPE,
                         list(index_values))
    size = destination_size if destination_size is not None \
        else len(index_values) * 3
    destination = typed_core((size,), dtype, FLOAT_SENTINEL[:size])
    return source, indices, destination


INDEX_SELECT_ABI_CASES = (
    ("null source", lambda lib, s, i, d: lib.tf_core_index_select(
        None, 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 2, 3, 3)),
    ("null index", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, None, 0,
        d._storage._require_open(), 1, 2, 3, 3)),
    ("null destination", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        None, 1, 2, 3, 3)),
    ("zero outer", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 0, 2, 3, 3)),
    ("negative outer", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), -1, 2, 3, 3)),
    ("zero axis_length", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 0, 3, 3)),
    ("negative axis_length", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, -2, 3, 3)),
    ("zero index_count", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 2, 0, 3)),
    ("negative index_count", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 2, -3, 3)),
    ("zero inner", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 2, 3, 0)),
    ("negative inner", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 2, 3, -3)),
    ("source product overflow", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), (2 ** 63 - 1) // 2, 4, 3, 1)),
    ("destination product overflow",
     lambda lib, s, i, d: lib.tf_core_index_select(
         s._storage._require_open(), 0, i._storage._require_open(), 0,
         d._storage._require_open(), 1, 2, (2 ** 63 - 1) // 2, 4)),
    ("negative source offset", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), -1, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 2, 3, 3)),
    ("negative index offset", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), -1,
        d._storage._require_open(), 1, 2, 3, 3)),
    ("source span too long", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 3, 3, 3)),
    ("source offset pushes the span out",
     lambda lib, s, i, d: lib.tf_core_index_select(
         s._storage._require_open(), 2, i._storage._require_open(), 0,
         d._storage._require_open(), 1, 2, 3, 3)),
    ("index span too long", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 1,
        d._storage._require_open(), 1, 2, 3, 3)),
    ("destination too small", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 2, 2, 3)),
    ("destination too large", lambda lib, s, i, d: lib.tf_core_index_select(
        s._storage._require_open(), 0, i._storage._require_open(), 0,
        d._storage._require_open(), 1, 2, 3, 2)),
)


@needs_native
@pytest.mark.parametrize("label, call", INDEX_SELECT_ABI_CASES,
                         ids=[row[0] for row in INDEX_SELECT_ABI_CASES])
def test_the_index_select_abi_rejects_malformed_metadata_and_writes_nothing(
        label, call, live_storages):
    """§22.9's list — its **own** list, longer than argmax's because it has
    three handles, two offsets, four extents, and two aliasing pairs."""
    library = cpp._require_library()
    library.tf_clear_error()
    source, indices, destination = _index_select_abi_fixture()
    baseline = settled(live_storages)
    try:
        source_before = raw_bytes(source)
        index_before = raw_bytes(indices)
        destination_before = raw_bytes(destination)
        with no_native_allocation():
            with pytest.raises(ValueError) as caught:
                call(library, source, indices, destination)
        assert "index_select" in str(caught.value)
        assert raw_bytes(destination) == destination_before, label
        assert raw_bytes(source) == source_before, label
        assert raw_bytes(indices) == index_before, label
        assert settled(live_storages) == baseline
        assert error_slot() == cpp.TF_OK
    finally:
        destination.close()
        indices.close()
        source.close()
        library.tf_clear_error()


# (label, source dtype, index dtype, destination dtype, message)
INDEX_SELECT_ROLE_REJECTIONS = (
    ("an int64 source", INDEX_DTYPE, INDEX_DTYPE, "float64", "floating"),
    ("an int64 source with a float32 destination",
     INDEX_DTYPE, INDEX_DTYPE, "float32", "floating"),
    ("an int64 destination", "float64", INDEX_DTYPE, INDEX_DTYPE, "floating"),
    ("an int64 destination with a float32 source",
     "float32", INDEX_DTYPE, INDEX_DTYPE, "floating"),
    ("a float64 source with a float32 destination",
     "float64", INDEX_DTYPE, "float32", "dtype"),
    ("a float32 source with a float64 destination",
     "float32", INDEX_DTYPE, "float64", "dtype"),
    ("a float64 index operand", "float64", "float64", "float64", "int64"),
    ("a float32 index operand", "float64", "float32", "float64", "int64"),
    ("a float32 index operand with a float32 value pair",
     "float32", "float32", "float32", "int64"),
)

_ROLE_VALUES = {
    "float64": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "float32": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    INDEX_DTYPE: [1, 2, 3, 4, 5, 6],
}


def _role_core(shape, dtype, count):
    """A prefilled core of ``count`` distinctive elements at ``dtype``, so
    a byte comparison after a rejection has something to see."""
    if dtype == INDEX_DTYPE:
        values = [(-11 - position) for position in range(count)]
    else:
        values = [(-1.5 - position) for position in range(count)]
    return typed_core(shape, dtype, values)


@needs_native
@pytest.mark.parametrize(
    "label, source_dtype, index_dtype, destination_dtype, message",
    INDEX_SELECT_ROLE_REJECTIONS,
    ids=[row[0] for row in INDEX_SELECT_ROLE_REJECTIONS])
def test_the_index_select_abi_rejects_every_dtype_role_error(
        label, source_dtype, index_dtype, destination_dtype, message,
        live_storages):
    """The role half of §22.9, the mirror image of argmax's and held to the
    same standard as the malformed metadata matrix: **all three** operands'
    bytes are captured and compared, no native storage is allocated, and
    the error slot is clean once Python's ``errcheck`` hook has handled the
    rejection.

    Both value handles must be floating **and matching**, and the index
    must be exactly ``int64``. ``require_matching_dtype`` is asked here and
    only here, and only across the floating pair — never across the
    floating/index boundary.
    """
    library = cpp._require_library()
    library.tf_clear_error()
    baseline = settled(live_storages)
    source = typed_core((6,), source_dtype, _ROLE_VALUES[source_dtype])
    indices = (typed_core((3,), index_dtype, [1, 0, 1])
               if index_dtype == INDEX_DTYPE
               else typed_core((3,), index_dtype, [1.0, 0.0, 1.0]))
    destination = _role_core((9,), destination_dtype, 9)
    try:
        source_before = raw_bytes(source)
        index_before = raw_bytes(indices)
        destination_before = raw_bytes(destination)
        with no_native_allocation():
            with pytest.raises(ValueError, match=message):
                library.tf_core_index_select(
                    source._storage._require_open(), 0,
                    indices._storage._require_open(), 0,
                    destination._storage._require_open(), 1, 2, 3, 3)
        assert raw_bytes(source) == source_before, label
        assert raw_bytes(indices) == index_before, label
        assert raw_bytes(destination) == destination_before, label
        assert settled(live_storages) == baseline + 3, label
        assert error_slot() == cpp.TF_OK, "the errcheck hook left a stale code"
    finally:
        destination.close()
        indices.close()
        source.close()
    assert settled(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_index_select_abi_accepts_the_one_valid_role_combination(
        dtype, live_storages):
    """The control the rejections above are measured against: a matching
    floating source/destination pair with an ``int64`` index really does
    execute, at both widths, and writes the selected slices."""
    library = cpp._require_library()
    library.tf_clear_error()
    baseline = settled(live_storages)
    source = typed_core((6,), dtype, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    indices = typed_core((3,), INDEX_DTYPE, [1, 0, 1])
    destination = _role_core((9,), dtype, 9)
    try:
        library.tf_core_index_select(
            source._storage._require_open(), 0,
            indices._storage._require_open(), 0,
            destination._storage._require_open(), 1, 2, 3, 3)
        assert same_bits(
            destination.to_numpy(),
            np.array([4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                     dtype=NUMPY_DTYPE[dtype]))
        assert source.dtype == destination.dtype != indices.dtype
        assert error_slot() == cpp.TF_OK
    finally:
        destination.close()
        indices.close()
        source.close()
    assert settled(live_storages) == baseline


@needs_native
def test_the_index_select_abi_rejects_both_aliasing_pairs(live_storages):
    """Two aliasing checks, because there are two operands. The
    destination/source pair is genuinely reachable — both are floating and
    must match — while the destination/index pair is refused by a role
    check first, and both are asserted to write nothing."""
    library = cpp._require_library()
    baseline = settled(live_storages)
    indices = typed_core((3,), INDEX_DTYPE, [1, 0, 1])
    shared = typed_core((9,), "float64", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                          7.0, 8.0, 9.0])
    try:
        before = raw_bytes(shared)
        with pytest.raises(ValueError, match="alias"):
            library.tf_core_index_select(
                shared._storage._require_open(), 0,
                indices._storage._require_open(), 0,
                shared._storage._require_open(), 1, 3, 3, 3)
        assert raw_bytes(shared) == before
        # destination aliasing the index: one storage carries one dtype, so
        # a role check answers first. What matters is that it is refused and
        # writes nothing.
        index_before = raw_bytes(indices)
        with pytest.raises(ValueError):
            library.tf_core_index_select(
                shared._storage._require_open(), 0,
                indices._storage._require_open(), 0,
                indices._storage._require_open(), 1, 3, 3, 1)
        assert raw_bytes(indices) == index_before
        assert raw_bytes(shared) == before
    finally:
        shared.close()
        indices.close()
    assert settled(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("bad", [-1, -9223372036854775808, 2, 99],
                         ids=["minus-one", "int64-min", "just-past", "far"])
def test_the_index_select_abi_rejects_every_out_of_range_index(bad,
                                                              live_storages):
    """Negative values are rejected rather than wrapped (§14.2), and an
    out-of-range value is rejected at every magnitude."""
    library = cpp._require_library()
    baseline = settled(live_storages)
    source, indices, destination = _index_select_abi_fixture(
        index_values=(0, 1, 0))
    here = settled(live_storages)
    try:
        indices._storage.copy_from(np.array([0, 1, bad], dtype=np.int64))
        before = raw_bytes(destination)
        with pytest.raises(ValueError) as caught:
            library.tf_core_index_select(
                source._storage._require_open(), 0,
                indices._storage._require_open(), 0,
                destination._storage._require_open(), 1, 2, 3, 3)
        message = str(caught.value)
        assert "index" in message and "position" in message
        assert raw_bytes(destination) == before
        assert settled(live_storages) == here
    finally:
        destination.close()
        indices.close()
        source.close()
    assert settled(live_storages) == baseline


# ===========================================================================
# 11. The complete index scan happens before any write
# ===========================================================================

@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_late_invalid_index_leaves_the_whole_destination_unchanged(
        dtype, live_storages):
    """§14.4 and §22.9 step 9, driven at the C ABI: several **valid**
    indices precede an invalid one, so an implementation that checked each
    index as it copied would have written a prefix before it threw.

    The source carries exceptional values on purpose — both signed zeros, an
    infinity, a subnormal, and a NaN — so a partial copy would be visible in
    raw bits even where a numeric comparison would not see it.
    """
    library = cpp._require_library()
    library.tf_clear_error()
    numpy_dtype = NUMPY_DTYPE[dtype]
    tiny = np.finfo(numpy_dtype).tiny
    payload = np.array([0.0, -0.0, np.inf, -np.inf, tiny / 4.0, np.nan,
                        1.0, 2.0], dtype=numpy_dtype)
    source = typed_core((8,), dtype)
    source._storage.copy_from(payload)
    # Four rows of two: axis_length 4, inner 2, outer 1.
    indices = typed_core((4,), INDEX_DTYPE, [0, 1, 2, 7])
    destination = typed_core((8,), dtype)
    sentinel = np.array([-11.5, -22.5, -33.5, -44.5, -55.5, -66.5,
                         -77.5, -88.5], dtype=numpy_dtype)
    destination._storage.copy_from(sentinel)
    baseline = settled(live_storages)
    try:
        before = raw_bytes(destination)
        assert before == sentinel.tobytes()
        with no_native_allocation():
            with pytest.raises(ValueError) as caught:
                library.tf_core_index_select(
                    source._storage._require_open(), 0,
                    indices._storage._require_open(), 0,
                    destination._storage._require_open(), 1, 4, 4, 2)
        message = str(caught.value)
        assert "7" in message and "3" in message      # value and position
        # Not one byte moved — not the prefix the three valid indices name,
        # and not any other byte either.
        assert raw_bytes(destination) == before
        assert bits(destination.to_numpy()) == bits(sentinel)
        assert raw_bytes(source) == payload.tobytes()
        assert settled(live_storages) == baseline
        assert error_slot() == cpp.TF_OK
        # The control: the same vector with the last index made valid writes
        # exactly the expected bits, so the no-write claim is not vacuous.
        indices._storage.copy_from(np.array([0, 1, 2, 3], dtype=np.int64))
        library.tf_core_index_select(
            source._storage._require_open(), 0,
            indices._storage._require_open(), 0,
            destination._storage._require_open(), 1, 4, 4, 2)
        assert bits(destination.to_numpy()) == bits(payload)
        assert raw_bytes(destination) != before
    finally:
        destination.close()
        indices.close()
        source.close()
        library.tf_clear_error()


@needs_native
def test_the_python_bounds_scan_is_complete_and_precedes_allocation(
        live_storages):
    """The Python half of the same rule, and it is a **second** authority
    rather than a restatement: a late invalid index rejects before the
    destination is allocated at all."""
    baseline = settled(live_storages)
    source = floating([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "float64")
    indices = index(np.array([0, 1, 2, 3], dtype=np.int64))
    here = settled(live_storages)
    try:
        with no_native_allocation():
            with pytest.raises(ValueError) as caught:
                source.index_select(0, indices)
        assert "3" in str(caught.value)
        assert settled(live_storages) == here
        # ...and the valid prefix alone succeeds.
        good = index(np.array([0, 1, 2], dtype=np.int64))
        try:
            result = source.index_select(0, good)
            try:
                assert same_bits(result.to_numpy(), source.to_numpy())
            finally:
                result.close()
        finally:
            good.close()
    finally:
        indices.close()
        source.close()
    assert settled(live_storages) == baseline


# ===========================================================================
# 12. Error-slot hygiene
# ===========================================================================

@needs_native
@pytest.mark.parametrize("export", INDEXING_EXPORTS)
def test_each_export_records_then_clears_the_thread_local_slot(export):
    """A guarded export records a rejection in the slot and clears it on the
    **next** entry, so one call's failure can never be misread as another's.

    Driven with the ``errcheck`` hook temporarily replaced by an identity,
    because that hook is what normally consumes and clears the slot; it is
    restored in a ``finally``.
    """
    library = cpp._require_library()
    function = getattr(library, export)
    hook = function.errcheck
    assert hook is not None, f"{export} is not hooked at all"
    source = typed_core((6,), "float64", [1.0, 7.0, 3.0, 9.0, 9.0, 2.0])
    indices = typed_core((3,), INDEX_DTYPE, [1, 0, 1])
    argmax_destination = typed_core((2,), INDEX_DTYPE, ARGMAX_SENTINEL)
    select_destination = typed_core((9,), "float64",
                                    FLOAT_SENTINEL[:9])
    try:
        function.errcheck = lambda result, callee, arguments: result
        library.tf_clear_error()
        if export == "tf_core_argmax":
            reject = lambda: function(source._storage._require_open(), 0,
                                      argmax_destination._storage
                                      ._require_open(), 0, 3, 1)
            accept = lambda: function(source._storage._require_open(), 0,
                                      argmax_destination._storage
                                      ._require_open(), 2, 3, 1)
        else:
            reject = lambda: function(
                source._storage._require_open(), 0,
                indices._storage._require_open(), 0,
                select_destination._storage._require_open(), 0, 2, 3, 3)
            accept = lambda: function(
                source._storage._require_open(), 0,
                indices._storage._require_open(), 0,
                select_destination._storage._require_open(), 1, 2, 3, 3)
        reject()
        assert library.tf_last_error_code() != cpp.TF_OK
        assert export.replace("tf_core_", "") in \
            library.tf_last_error_message().decode()
        # A valid call clears the slot on **entry** and succeeds.
        accept()
        assert library.tf_last_error_code() == cpp.TF_OK
    finally:
        function.errcheck = hook
        library.tf_clear_error()
        select_destination.close()
        argmax_destination.close()
        indices.close()
        source.close()
    # ...and with the hook restored a rejection is a Python exception again,
    # and it leaves no code behind for the next call.
    assert error_slot() == cpp.TF_OK


@needs_fault_injection
def test_a_disarmed_allocation_failure_leaves_no_stale_error(live_storages):
    """One test's injected failure cannot change another test's result: the
    slot is clear after the arm is taken back out, and the next valid call
    succeeds from a clean slot."""
    baseline = settled(live_storages)
    with arm_native_allocation(lambda size, dtype: dtype == INDEX_DTYPE):
        with pytest.raises(MemoryError):
            index(np.array([1, 2], dtype=np.int64))
    assert error_slot() == cpp.TF_OK, "a stale native error survived"
    recovered = index(np.array([1, 2], dtype=np.int64))
    try:
        assert exact_ints(recovered.to_numpy()) == [1, 2]
        assert error_slot() == cpp.TF_OK
    finally:
        recovered.close()
    assert settled(live_storages) == baseline


@needs_native
def test_the_unhooked_scalar_primitives_gained_no_error_slot_behavior():
    """``tf_storage_fill`` and ``tf_storage_scale`` are **guarded but
    unhooked**, and the two words mean different things (§22.5 and the
    ``_CHECKED_KERNELS`` note).

    K1 gave both ``TF_GUARD_BEGIN`` / ``TF_GUARD_END_VOID`` precisely so
    they could *record* an integer-role rejection instead of letting an
    exception escape the C ABI — so calling them "unguarded" would describe
    the opposite of what K1 did. What they deliberately are **not** is
    members of ``_CHECKED_KERNELS``: no Python ``errcheck`` hook is
    attached, so a rejection they record is readable in the thread-local
    slot rather than raised, which is what keeps every fill and every mean
    one native call rather than two. The Python wrapper is the layer that
    refuses the unsupported ``int64`` role, ahead of the native call
    entirely.
    """
    library = cpp._require_library()
    assert "tf_storage_fill" not in cpp._CHECKED_KERNELS
    assert "tf_storage_scale" not in cpp._CHECKED_KERNELS
    assert getattr(library.tf_storage_fill, "errcheck", None) in (None, )
    assert getattr(library.tf_storage_scale, "errcheck", None) in (None, )
    # "Guarded" is a structural claim about the C++ source, so it is read
    # from the source with comments and string literals stripped, not
    # asserted from the Python side where it would be invisible.
    storage_source = cpp_code_only(
        (REPO_ROOT / "cpp" / "src" / "storage.cpp").read_text(encoding="utf-8"))
    for export in ("tf_storage_fill", "tf_storage_scale"):
        body = re.search(
            rf"TF_EXPORT\s+void\s+{export}\s*\([^)]*\)\s*\{{(.*?)\n\}}",
            storage_source, re.S)
        assert body is not None, export
        assert "TF_GUARD_BEGIN" in body.group(1), export
        assert "TF_GUARD_END_VOID" in body.group(1), export
        assert "require_floating" in body.group(1), export
    library.tf_clear_error()
    integer = typed_core((2,), INDEX_DTYPE, [3, 4])
    try:
        before = raw_bytes(integer)
        library.tf_storage_fill(integer._storage._require_open(), 1.0)
        assert library.tf_last_error_code() != cpp.TF_OK
        assert raw_bytes(integer) == before, "a rejected fill wrote anyway"
    finally:
        library.tf_clear_error()
        integer.close()
    # ...and the Python wrapper refuses it before the native call at all.
    held = index(np.array([3, 4], dtype=np.int64))
    try:
        with pytest.raises(ValueError):
            held._core.storage.fill(1.0)
    finally:
        held.close()
    assert error_slot() == cpp.TF_OK


# ===========================================================================
# 13. Parsers and scanners — each with a planted in-memory control
# ===========================================================================

def code_identifiers(source):
    """Every identifier a module's **executable code** names.

    A source-text scan would be wrong: this repository's modules explain at
    length what they deliberately do *not* do, so a prose mention of
    ``threading`` would fail a substring check that is supposed to be about
    behavior. The AST asks the question that was meant, and it reads
    keyword-argument names and imports too.
    """
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.add((node.module or "").split(".")[0])
            for alias in node.names:
                names.add(alias.name)
    return names


def cpp_code_only(source):
    """C++ source with comments and string literals removed, so a scan of
    validation *structure* cannot be satisfied or broken by prose."""
    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    without_line = re.sub(r"//[^\n]*", " ", without_block)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', without_line)


def test_the_identifier_scanner_can_actually_fail():
    """Planted in-memory controls: prose must not leak into the code view,
    and keyword arguments and imports must be visible to it."""
    names = code_identifiers(
        '"""a docstring naming threading and gather."""\n'
        'import os\n'
        'def f(x):\n'
        '    return g(x, dtype="int64", _trusted_dtype=True)\n')
    assert "threading" not in names, "prose leaked into the code view"
    assert "gather" not in names
    assert {"os", "f", "g", "x", "dtype", "_trusted_dtype"} <= names
    # ...and it does see a real import and a real call.
    planted = code_identifiers("import threading\nthreading.Thread()\n")
    assert "threading" in planted and "Thread" in planted


def test_the_cpp_comment_stripper_can_actually_fail():
    stripped = cpp_code_only(
        "// require_floating is not applied here\n"
        "/* require_matching_dtype either */\n"
        'const char* m = "require_index";\n'
        "if (!tf::require_floating(op, {src})) { return; }\n")
    assert "require_matching_dtype" not in stripped
    assert stripped.count("require_floating") == 1, stripped
    assert "require_index" not in stripped


def test_the_export_scanner_can_actually_fail():
    """The inventory parser must find a planted extra export and must not
    be fooled by one that only appears in a comment."""
    pattern = r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\("
    planted = cpp_code_only(
        "// TF_EXPORT void tf_core_ghost(void);\n"
        "TF_EXPORT void tf_core_real(void) {}\n"
        "TF_EXPORT void tf_core_extra(void) {}\n")
    found = set(re.findall(pattern, planted, re.S))
    assert found == {"tf_core_real", "tf_core_extra"}, found
    assert "tf_core_ghost" not in found


def test_the_guard_order_scanner_can_actually_fail():
    """Where a guard **order** is contractual (§22.9 steps 2-5), the scanner
    must notice a reordering and a missing guard, on planted strings."""
    def guard_order(source):
        code = cpp_code_only(source)
        return [match.group(1) for match in re.finditer(
            r"tf::(require_floating|require_matching_dtype|require_index)",
            code)]

    correct = ("if (!tf::require_floating(op, {src})) return;\n"
               "if (!tf::require_floating(op, {dst})) return;\n"
               "if (!tf::require_matching_dtype(op, src, dst)) return;\n"
               "if (!tf::require_index(op, \"index\", idx)) return;\n")
    assert guard_order(correct) == ["require_floating", "require_floating",
                                    "require_matching_dtype", "require_index"]
    swapped = ("if (!tf::require_matching_dtype(op, src, dst)) return;\n"
               "if (!tf::require_floating(op, {src})) return;\n")
    assert guard_order(swapped) != guard_order(correct)
    missing = "if (!tf::require_floating(op, {src})) return;\n"
    assert "require_index" not in guard_order(missing)


@needs_native
def test_the_live_export_guard_order_is_the_contracted_one():
    """The real source, read through the stripped code view: each export
    applies the guards §22.8 and §22.9 give it, in that order, and neither
    applies one the other forbids."""
    code = cpp_code_only(
        (REPO_ROOT / "cpp" / "src" / "indexing.cpp").read_text(
            encoding="utf-8"))
    argmax_body = code.split("TF_EXPORT void tf_core_argmax(", 1)[1] \
        .split("TF_EXPORT void tf_core_index_select(", 1)[0]
    select_body = code.split("TF_EXPORT void tf_core_index_select(", 1)[1]
    argmax_guards = re.findall(
        r"tf::(require_floating|require_matching_dtype|require_index)",
        argmax_body)
    select_guards = re.findall(
        r"tf::(require_floating|require_matching_dtype|require_index)",
        select_body)
    # argmax: a floating source and an int64 destination, and **neither**
    # require_floating nor require_matching_dtype on that destination.
    assert argmax_guards == ["require_floating", "require_index"], \
        argmax_guards
    # index_select: both value handles floating, then matching, then the
    # index role — the one place require_matching_dtype is used at all.
    assert select_guards == ["require_floating", "require_floating",
                             "require_matching_dtype", "require_index"], \
        select_guards
    assert argmax_guards.count("require_matching_dtype") == 0
    # ...and neither switch has a default arm that could silently misread a
    # future dtype.
    assert "default:" not in argmax_body and "default:" not in select_body


@needs_native
def test_the_two_exports_do_not_share_an_argument_validator():
    """§22.10: one blanket validator would need a mode flag, and a mode flag
    is how two contracts quietly become one."""
    code = cpp_code_only(
        (REPO_ROOT / "cpp" / "src" / "indexing.cpp").read_text(
            encoding="utf-8"))
    assert "const char* argument_error(" in code
    assert "const char* index_select_argument_error(" in code
    argmax_body = code.split("TF_EXPORT void tf_core_argmax(", 1)[1] \
        .split("TF_EXPORT void tf_core_index_select(", 1)[0]
    select_body = code.split("TF_EXPORT void tf_core_index_select(", 1)[1]
    assert "index_select_argument_error(" not in argmax_body
    assert re.search(r"[^_]argument_error\(", argmax_body)
    assert "index_select_argument_error(" in select_body


# ===========================================================================
# 14. Concurrency, stable/native isolation, and the inventories
# ===========================================================================

def test_this_module_starts_no_thread_and_names_no_concurrency_machinery():
    """§25: concurrency is a documented boundary, never a tested safety
    claim. Read from this module's own AST, so a prose mention of
    ``threading`` in the docstring above cannot fail it — and a real import
    could not pass."""
    names = code_identifiers(THIS_MODULE.read_text(encoding="utf-8"))
    for banned in CONCURRENCY_NAMES:
        assert banned not in names, banned
    # The planted control, in memory only: the same scan on a mutated copy
    # of this module must fail.
    mutated = THIS_MODULE.read_text(encoding="utf-8") + \
        "\nimport threading\n"
    assert "threading" in code_identifiers(mutated)


@needs_native
def test_importing_the_stable_framework_loads_no_native_library():
    """§24, re-proved after every injection above: the stable line never
    pulls in ctypes or the C++ library, and no environment variable changes
    which line runs. Run in a subprocess because this process has already
    imported everything."""
    program = (
        "import sys, tensorforge\n"
        "assert 'tensorforge.backends.cpp' not in sys.modules\n"
        "from tensorforge.backends import cpp\n"
        "assert cpp._lib is None, 'importing tensorforge loaded the library'\n"
        "assert cpp.SUPPORTED_DTYPES == ('float64', 'float32')\n"
        "assert cpp.INDEX_DTYPES == ('int64',)\n"
        "assert cpp.backend_info()['stable_framework_integration'] is False\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


@needs_native
def test_stable_and_native_objects_reject_one_another_across_the_phase():
    """No stable object reaches integer ingress and no native integer tensor
    reaches a stable API, in either direction — and neither leaves anything
    behind."""
    stable = tensorforge.Tensor([[1.0, 2.0]])
    held = index(np.array([1, 0], dtype=np.int64))
    source = floating([[1.0, 2.0], [3.0, 4.0]], "float64")
    try:
        with no_native_allocation():
            with pytest.raises((TypeError, ValueError)):
                NativeTensor.from_int64_array(stable)
            with pytest.raises((TypeError, ValueError)):
                source.index_select(0, stable)
            with pytest.raises((TypeError, ValueError)):
                stable + held
            with pytest.raises((TypeError, ValueError)):
                NativeParameter(held)
            with pytest.raises((TypeError, ValueError)):
                held.argmax()
            with pytest.raises((TypeError, ValueError)):
                source.add(held)
    finally:
        source.close()
        held.close()
    assert cpp.backend_info()["stable_framework_integration"] is False


def test_no_environment_variable_selects_a_dtype_a_backend_or_a_path():
    source = inspect.getsource(cpp)
    for banned in ("os.environ", "getenv", "TF_DTYPE", "TENSORFORGE_DTYPE"):
        assert banned not in source, banned


@needs_native
def test_every_inventory_is_exactly_what_k6_left():
    """K7 is a test-and-status milestone: nothing here may move a count."""
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.INDEX_DTYPES == ("int64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")
    assert len(cpp._CHECKED_KERNELS) == K7_CHECKED_KERNELS
    assert len(experimental.__all__) == K7_EXPERIMENTAL_EXPORTS
    assert "argmax" in cpp.TENSOR_CORE_OPS
    assert "index_select" in cpp.TENSOR_CORE_OPS
    assert "argmax" not in cpp.AUTOGRAD_OPS
    assert "index_select" not in cpp.AUTOGRAD_OPS
    assert checkpoint_module._FORMAT_VERSION == 3
    assert checkpoint_module._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert optimizer_state.FORMAT_VERSION == 1
    assert loader_module._FORMAT_VERSION == 1
    assert sampler_module._FORMAT_VERSION == 1
    assert len(list((REPO_ROOT / "examples").glob("*.py"))) == K7_EXAMPLE_COUNT
    benchmarks = [path.name
                  for path in (REPO_ROOT / "benchmarks").glob("*.py")]
    assert len(benchmarks) == K7_BENCHMARK_COUNT, sorted(benchmarks)
    for name, milestone in POST_K7_BENCHMARKS.items():
        assert name in benchmarks, (name, milestone)
    assert len([name for name in benchmarks
                if name not in POST_K7_BENCHMARKS]) == K7_OWN_BENCHMARK_COUNT


@needs_native
def test_the_source_and_built_export_inventories_agree_at_fifty_six():
    """Neither K7 nor any injection above added or removed a symbol, and the
    source inventory and the loaded library still agree exactly."""
    pattern = r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\("
    source_exports = set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        source_exports.update(re.findall(
            pattern, cpp_code_only(path.read_text(encoding="utf-8")), re.S))
    assert len(source_exports) == K7_EXPORT_COUNT, sorted(source_exports)
    for export in INDEXING_EXPORTS:
        assert export in source_exports
    # The symbols K7 must not have invented, and the ones a later phase
    # would need — none of them exists.
    for absent in ("tf_core_gather", "tf_core_scatter", "tf_core_scatter_add",
                   "tf_core_embedding", "tf_core_max", "tf_core_argmin",
                   "tf_core_max_with_indices", "tf_core_argmax_backward",
                   "tf_core_index_select_backward", "tf_storage_dtype"):
        assert absent not in source_exports, absent
    library = cpp._require_library()
    for name in sorted(source_exports):
        assert hasattr(library, name), f"{name} is not in the built library"
    # ...and the built image's own export table, read from the binary, is
    # exactly that set — the stale-artifact guard. Imported defensively
    # rather than through ``importorskip`` so no path here can produce a
    # skip; when the image format is not parsed on this platform the
    # ``hasattr`` sweep above is still a real source/built reconciliation.
    try:
        import test_native_storage_allocation as storage_tests
    except ImportError:                                   # pragma: no cover
        storage_tests = None
    if storage_tests is not None:
        _, names = storage_tests.exported_names(cpp._LIBRARY_PATH)
        if names is not None:
            exported = {name for name in names if name.startswith("tf_")}
            assert exported == source_exports, \
                sorted(exported ^ source_exports)


@needs_native
def test_the_ctest_inventory_did_not_move():
    text = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    registered = re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", text)
    assert len(registered) == K7_CTEST_COUNT, registered
    assert len(set(registered)) == len(registered)
    for expected in ("argmax", "index_select"):
        assert expected in registered, expected


def test_no_k9_artifact_appeared():
    """K7 is not K9: the closure module must still be absent.

    Through K7 this guard also asserted K8's benchmark and its owner
    absent. That premise expired when **K8** landed, so the two entries
    moved from the absent list to a present one rather than being deleted
    — a guard that keeps banning a file the repository now legitimately
    owns is a guard that forces a lie, and one that simply drops the entry
    stops being a claim about the ladder at all."""
    for relative in K8_ARTIFACTS:
        assert (REPO_ROOT / relative).is_file(), relative
    for relative in K9_ARTIFACTS:
        assert not (REPO_ROOT / relative).exists(), relative


def test_this_module_asserts_no_timing_and_no_tolerance():
    """The discipline this file claims for itself, checked against its own
    executable code rather than its prose."""
    names = code_identifiers(THIS_MODULE.read_text(encoding="utf-8"))
    for banned in ("allclose", "approx", "isclose", "perf_counter",
                   "monotonic", "process_time"):
        assert banned not in names, banned


def _function_bodies(source):
    """Every top-level and nested function definition in ``source``, by
    name, as AST nodes."""
    bodies = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies[node.name] = node
    return bodies


def _called_names(node):
    """Every callee name reached from ``node``, attribute or plain."""
    found = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            function = inner.func
            if isinstance(function, ast.Name):
                found.add(function.id)
            elif isinstance(function, ast.Attribute):
                found.add(function.attr)
    return found


def test_every_injection_matrix_owner_carries_the_complete_world_fingerprint():
    """The claim "every rejection and every injected failure is surrounded
    by the complete observable-world fingerprint" is made **literal** here
    rather than left as prose: every non-``N/A`` row of
    ``INJECTION_MATRIX`` is read from this module's own AST and must
    actually enter ``unchanged_world``.

    A narrower check — a live-storage baseline, or ``no_native_allocation``
    alone — is a real assertion but a different one, and it may not be
    reported as the complete fingerprint.
    """
    source = THIS_MODULE.read_text(encoding="utf-8")
    bodies = _function_bodies(source)
    narrower = []
    for operation, position, _family, owner in INJECTION_MATRIX:
        if owner.startswith("N/A:"):
            continue
        assert owner in bodies, owner
        if "unchanged_world" not in _called_names(bodies[owner]):
            narrower.append((operation, position, owner))
    assert not narrower, narrower

    # Non-vacuity, planted **in memory** — the repository file is never
    # edited to make a scanner fail.
    planted = source + (
        "\n\ndef _planted_narrow_owner():\n"
        "    assert settled(None) == 0\n"
    )
    planted_bodies = _function_bodies(planted)
    assert "unchanged_world" not in _called_names(
        planted_bodies["_planted_narrow_owner"])
    assert "unchanged_world" in _called_names(
        planted_bodies[INJECTION_MATRIX[0][3]])


def test_this_module_names_every_caller_owned_native_object_it_builds():
    """No caller-owned native object is constructed inline as a call
    argument, because nothing would then hold it and nothing could close
    it — and §9 forbids any assertion here from resting on ``__del__``.

    The one permitted shape is an **ownership transfer**: a buffer handed
    to ``register_buffer`` belongs to the module afterwards and is closed
    through ``named_buffers()`` in ``Sentinels.close``, so it is named in
    an allow-list with that reason rather than silently ignored.
    """
    owning = {"index", "floating", "typed_core", "strided_view",
              "_role_core", "from_int64_array", "from_array",
              "contiguous_copy", "index_select", "argmax"}
    adopting = {"register_buffer"}          # transfers ownership, see above
    source = THIS_MODULE.read_text(encoding="utf-8")

    def inline_constructions(text):
        found = []
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            outer = (node.func.id if isinstance(node.func, ast.Name)
                     else node.func.attr
                     if isinstance(node.func, ast.Attribute) else None)
            if outer in adopting:
                continue
            arguments = list(node.args) + [kw.value for kw in node.keywords]
            for argument in arguments:
                if not isinstance(argument, ast.Call):
                    continue
                inner = (argument.func.id
                         if isinstance(argument.func, ast.Name)
                         else argument.func.attr
                         if isinstance(argument.func, ast.Attribute) else None)
                if inner in owning:
                    found.append((argument.lineno, outer, inner))
        return found

    assert inline_constructions(source) == []

    # Non-vacuity, planted in memory: the scanner sees the shape it bans.
    planted = source + (
        "\n\ndef _planted_inline_owner(src):\n"
        "    return src.index_select(0, index([1, 0]))\n"
    )
    assert len(inline_constructions(planted)) == 1
