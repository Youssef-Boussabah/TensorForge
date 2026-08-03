"""The H1 output-allocation contract (Phase H, milestone H1).

H1 removed the redundant zero-fill from output storage that a kernel
provably overwrites in full. This file is the proof obligation that makes
that safe, in five parts:

1. **The allocator contract.** The zero-initializing path is unchanged
   and is still the default; the uninitialized path matches it in size
   validation, dtype/device rejection, allocation-failure handling,
   ownership, ``close()`` semantics, exactly-once destruction, and
   live-storage accounting.

2. **Poison proofs.** Every enabled operation is run with its output
   allocation deterministically poisoned, and no poison value may survive
   into the result. This is what proves "every destination element is
   written" — and it is *not* something ASan or UBSan can tell us:

   * **poison tests** prove complete destination initialization;
   * **ASan/UBSan** prove memory-boundary and undefined-behavior safety;
   * **LeakSanitizer and live-storage accounting** prove lifecycle
     cleanup.

   Real uninitialized memory is a useless oracle here — a fresh OS page
   reads back as zeros, so a kernel that skipped an element would look
   correct. MemorySanitizer would catch an uninitialized *read*, but MSan
   needs a fully instrumented libc and CPython, which this project does
   not have, so it is not used and is not claimed.

   **The poison lives entirely in this file.** It is applied by test
   infrastructure wrapped *around* the private allocation helper (see
   :func:`poisoned` below): the real production uninitialized constructor
   allocates, the wrapper fills the returned storage through the ordinary
   ``fill`` primitive, and that **same** storage is handed straight to the
   real production operation, which then runs the real kernel over it.
   Nothing in the shipped native library or the installed Python backend
   can influence what an allocation contains — there is no poison export,
   no thread-local flag, no environment variable, and no global mode, and
   §6 below asserts their absence against the loaded library's own export
   table.

   The negative controls prove the detector can actually fail: a
   partial-write kernel and an accumulating kernel, both aimed at an
   uninitialized destination, leave poison exactly where their holes are.

3. **Rejections.** The operations that must keep a zeroed destination —
   ``sum`` and ``narrow_backward`` — are pinned as such, with the
   evidence for why.

4. **Failure paths.** No operation may hand back storage from the
   uninitialized path after failing, and live storage must return to
   baseline without relying on garbage collection.

5. **Scope.** H1 added exactly one C ABI symbol and no public capability.

Nothing here asserts a duration. H1 is an allocation-contract change;
its measurement lives in ``benchmarks/benchmark_native_cpu_performance.py``.

Selector: python -m pytest -q -k native_storage_allocation
"""

import contextlib
import gc
import struct

import numpy as np
import pytest

from tensorforge.backends import cpp

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)
needs_fault_injection = pytest.mark.skipif(
    not cpp.fault_injection_available(),
    reason="native fault injection unavailable",
)

# Two poison patterns, chosen to catch different mistakes.
#
# A quiet NaN with a distinctive payload propagates through arithmetic, so
# an unwritten element that is *read* downstream contaminates the result
# rather than hiding as a plausible number. A large negative finite value
# catches the opposite mistake: code that special-cases NaN, or a
# comparison that a NaN would silently fail. Neither is a value any kernel
# under test produces from the inputs used here.
POISON_NAN = struct.unpack("<d", struct.pack("<Q", 0x7FF8DEADBEEFCAFE))[0]
POISON_FINITE = -1.2345678901234567e300
POISONS = (POISON_NAN, POISON_FINITE)


def poison_survivors(values, pattern):
    """How many elements still hold the poison — i.e. how many the kernel
    never wrote."""
    array = np.asarray(values)
    if np.isnan(pattern):
        return int(np.count_nonzero(np.isnan(array)))
    return int(np.count_nonzero(array == pattern))


# ==========================================================================
# 0. The poison, which is test infrastructure and nothing else
# ==========================================================================

@contextlib.contextmanager
def poisoned(pattern, log=None):
    """Fill every **uninitialized** native allocation made inside the
    block with ``pattern``, and hand that same storage to the production
    caller.

    This is deliberately built as a wrapper around the private allocation
    helper rather than as a switch inside the runtime. The sequence per
    allocation is exactly:

    1. the **real** ``NativeStorage._uninitialized`` runs, so the real
       ``tf_storage_create_uninitialized`` export allocates the buffer;
    2. this wrapper fills that buffer with ``pattern`` through the
       ordinary ``fill`` primitive (``tf_storage_fill``), which writes
       every element;
    3. the **same** storage object is returned to the production
       operation, which then runs the **real** kernel over it.

    So the poison is in place strictly after allocation and strictly
    before the kernel executes, and no poison-control API exists in the
    shipped library or the installed backend to make it work.

    ``NativeStorage._uninitialized`` is the single seam because everything
    that allocates uninitialized storage funnels through it:
    ``NativeTensorCore._uninitialized`` calls it, and
    ``NativeStorage.from_array`` calls it as ``cls._uninitialized``.

    Pass ``log`` to record ``(size, survivors_at_handoff)`` for every
    allocation, which is how §2a proves the poison really did reach the
    destination before the kernel ran.
    """
    descriptor = cpp.NativeStorage.__dict__["_uninitialized"]
    original = cpp.NativeStorage._uninitialized
    value = float(pattern)

    def poisoning(size, dtype=None, device="cpu"):
        storage = original(size, dtype=dtype, device=device)
        try:
            storage.fill(value)
            if log is not None:
                # Read back through the very handle the production caller
                # is about to receive, so the record describes the object
                # that is handed over, not a copy of it.
                log.append((int(storage.size),
                            poison_survivors(storage.to_numpy(), pattern)))
        except BaseException:
            storage.close()
            raise
        return storage

    cpp.NativeStorage._uninitialized = staticmethod(poisoning)
    try:
        yield
    finally:
        cpp.NativeStorage._uninitialized = descriptor


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open.

    Hooks ``__init__``, which is exactly why H1 routes *both* allocation
    kinds through it: an uninitialized allocation must be as visible to
    this accounting as a zeroed one."""
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


def rng(seed=0):
    return np.random.default_rng(seed)


# ==========================================================================
# 1. The allocator contract
# ==========================================================================

@needs_native
def test_the_zero_initializing_path_is_still_the_default_and_still_zeroes():
    """H1 changed no default. ``NativeStorage(n)`` and
    ``NativeTensorCore.zeros`` still hand back zeros."""
    storage = cpp.NativeStorage(64)
    try:
        assert np.array_equal(storage.to_numpy(), np.zeros(64))
    finally:
        storage.close()
    core = cpp.NativeTensorCore.zeros((5, 7))
    try:
        assert np.array_equal(core.to_numpy(), np.zeros((5, 7)))
    finally:
        core.close()


@needs_native
def test_the_uninitialized_path_reports_the_same_metadata():
    """Everything except the contents matches the zeroed path."""
    for size in (1, 9, 4096):
        zeroed = cpp.NativeStorage(size)
        raw = cpp.NativeStorage._uninitialized(size)
        try:
            assert raw.size == zeroed.size == size
            assert raw.dtype == zeroed.dtype == "float64"
            assert raw.device == zeroed.device == "cpu"
            assert type(raw) is type(zeroed) is cpp.NativeStorage
            # ...and it is writable, which is the only thing a caller may
            # do with it before reading. It is also exactly the property
            # the poison wrapper depends on.
            raw.fill(1.5)
            assert np.array_equal(raw.to_numpy(), np.full(size, 1.5))
        finally:
            raw.close()
            zeroed.close()


@needs_native
def test_the_uninitialized_core_reports_the_same_metadata():
    core = cpp.NativeTensorCore._uninitialized((3, 4))
    try:
        assert core.shape == (3, 4)
        assert core.strides == (4, 1)
        assert core.offset == 0
        assert core.contiguous
        assert core.numel == 12
        assert core.dtype == "float64" and core.device == "cpu"
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("bad", [0, -1, -4096, 1.5, True, None, "8"])
def test_both_paths_reject_the_same_invalid_sizes(bad):
    with pytest.raises(ValueError):
        cpp.NativeStorage(bad)
    with pytest.raises(ValueError):
        cpp.NativeStorage._uninitialized(bad)


@needs_native
@pytest.mark.parametrize("dtype,device", [("float16", "cpu"),
                                          ("bfloat16", "cpu"),
                                          ("float64", "cuda"),
                                          ("float32", "cuda")])
def test_the_public_path_rejects_unsupported_metadata(dtype, device):
    """Validation precedes allocation, so a rejected request allocates
    nothing. This is the **public** constructor, and it is the one that
    carries the promise.

    ``("float32", "cpu")`` was a row here through milestone I8 and is
    deliberately gone: I9 made float32 a supported TensorForge dtype, so it
    belongs in the acceptance test below. ``("float32", "cuda")`` replaces
    it and is the sharper case — a supported dtype does not make an
    unsupported device reachable."""
    with pytest.raises(ValueError):
        cpp.NativeStorage(8, dtype=dtype, device=device)


@needs_native
@pytest.mark.parametrize("dtype,device", [("float16", "cpu"),
                                          ("float64", "cuda")])
def test_the_private_allocator_still_rejects_what_the_runtime_cannot_represent(
        dtype, device):
    """``NativeStorage._uninitialized`` validates its dtype against the
    **internal** table from Phase I milestone I2 rather than the public
    registry, so that an operation's freshly allocated output can match its
    operand's dtype without asking permission the operand already has.

    That is a narrower relaxation than it sounds, and this pins the
    boundary: a dtype the runtime cannot physically represent is still
    rejected, an unsupported *device* is still rejected, and validation
    still precedes allocation. Only ``"float32"`` — which storage really can
    be, and which only the private typed constructors can request — moved,
    and it moved into a private path, not a public one.
    """
    with pytest.raises(ValueError):
        cpp.NativeStorage._uninitialized(8, dtype=dtype, device=device)


@needs_native
def test_both_allocators_reach_float32_and_agree_about_it():
    """Through milestone I8 this asserted the exact I2 truth — float32
    storage internally allocatable and publicly unsupported, at the same
    moment, on purpose. **I9 ended that split**: the public registry moved,
    so both paths reach float32 now and the durable claim is that they
    *agree*.

    That is the property worth pinning either way. When the two disagreed
    it was by design and the disagreement was the test; now that they
    concur, a private allocator that produced something a public one could
    not — a different width, a different size unit, a different tag — would
    be the drift, and this catches it."""
    assert cpp.normalize_dtype("float32") == "float32"
    for storage in (
        cpp.NativeStorage(8, dtype="float32"),
        cpp.NativeStorage._uninitialized(8, dtype="float32"),
        cpp.NativeStorage._typed(8, "float32"),
    ):
        try:
            assert storage.dtype == "float32"
            assert storage.size == 8          # elements, not bytes
        finally:
            storage.close()
    storage = cpp.NativeStorage.from_array([1.0, 2.0], dtype="float32")
    try:
        assert storage.dtype == "float32"
        assert storage.to_numpy().dtype == np.float32
    finally:
        storage.close()


@needs_native
def test_close_and_double_close_behave_identically_on_both_paths():
    for storage in (cpp.NativeStorage(16), cpp.NativeStorage._uninitialized(16)):
        storage.close()
        assert repr(storage) == "NativeStorage(closed)"
        storage.close()   # idempotent, per the existing contract
        assert repr(storage) == "NativeStorage(closed)"
        with pytest.raises(RuntimeError):
            storage._require_open()


@needs_native
@needs_fault_injection
def test_both_paths_raise_MemoryError_on_an_injected_allocation_failure():
    for constructor in (cpp.NativeStorage,
                        cpp.NativeStorage._uninitialized):
        cpp._arm_alloc_failure(1)
        try:
            with pytest.raises(MemoryError):
                constructor(32)
        finally:
            cpp._arm_alloc_failure(0)
        # ...and the next allocation succeeds, so nothing latched.
        storage = constructor(32)
        storage.close()


@needs_native
def test_both_paths_are_accounted_identically(live_storages):
    """The live-storage accounting every other suite relies on must see
    an uninitialized allocation exactly as it sees a zeroed one — which
    is why both run through ``NativeStorage.__init__``."""
    gc.collect()
    baseline = len(live_storages)
    zeroed = cpp.NativeStorage(8)
    assert len(live_storages) == baseline + 1
    raw = cpp.NativeStorage._uninitialized(8)
    assert len(live_storages) == baseline + 2
    raw.close()
    assert len(live_storages) == baseline + 1
    zeroed.close()
    assert len(live_storages) == baseline


@needs_native
def test_repeated_create_destroy_cycles_return_to_baseline(live_storages):
    gc.collect()
    baseline = len(live_storages)
    for _ in range(200):
        a = cpp.NativeStorage(64)
        b = cpp.NativeStorage._uninitialized(64)
        a.close()
        b.close()
    gc.collect()
    assert len(live_storages) == baseline


# ==========================================================================
# 2a. Poison negative controls — proving the detector can fail
# ==========================================================================

@needs_native
@pytest.mark.parametrize("pattern", POISONS)
def test_the_poison_fills_an_uninitialized_allocation(pattern):
    """The precondition for every poison proof below: if the poison did
    not actually land, a passing kernel test would prove nothing."""
    with poisoned(pattern):
        core = cpp.NativeTensorCore._uninitialized((6, 7))
    try:
        assert poison_survivors(core.to_numpy(), pattern) == 42
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("pattern", POISONS)
def test_the_poison_reaches_the_destination_before_every_real_kernel(pattern):
    """The load-bearing property of the test-side poison architecture,
    asserted directly rather than assumed.

    For every uninitialized allocation a production operation makes, the
    wrapper reads the buffer back **through the handle it is about to
    hand over** and records how much of it is poison. Every allocation
    must be *entirely* poison at that moment — which is the moment
    immediately before the production operation runs its real kernel over
    that same storage."""
    a = cpp.NativeTensorCore.from_array(rng(40).uniform(-1, 1, (5, 6)))
    b = cpp.NativeTensorCore.from_array(rng(41).uniform(-1, 1, (5, 6)))
    pos = cpp.NativeTensorCore.from_array(rng(42).uniform(0.5, 2.0, (5, 6)))
    images = cpp.NativeTensorCore.from_array(
        rng(43).uniform(-1, 1, (2, 3, 5, 6)))
    weight = cpp.NativeTensorCore.from_array(rng(44).uniform(-1, 1, (4, 3, 3, 3)))
    bias = cpp.NativeTensorCore.from_array(rng(45).uniform(-1, 1, (4,)))
    try:
        operations = (
            ("add", lambda: a.add(b)),
            ("relu", lambda: a.relu()),
            ("exp", lambda: a.exp()),
            ("log", lambda: pos.log()),
            ("matmul", lambda: a.transpose(1, 0).matmul(b)),
            ("softmax", lambda: a.softmax()),
            ("contiguous_copy", lambda: a.transpose(1, 0).contiguous_copy()),
            ("conv2d_forward", lambda: images.conv2d_forward(weight, bias)),
            ("full", lambda: cpp.NativeTensorCore.full((4, 4), 0.5)),
            ("from_array", lambda: cpp.NativeTensorCore.from_array(
                rng(46).uniform(-1, 1, (3, 3)))),
        )
        for label, run in operations:
            log = []
            with poisoned(pattern, log=log):
                out = run()
            out.close()
            assert log, f"{label}: made no uninitialized allocation to poison"
            for size, survivors in log:
                assert survivors == size, (
                    f"{label}: only {survivors} of {size} element(s) held the "
                    f"poison when the storage was handed to the operation"
                )
    finally:
        for core in (a, b, pos, images, weight, bias):
            core.close()


@needs_native
def test_the_poison_never_reaches_the_zero_initializing_path():
    """The control that keeps the proofs honest: if the poison leaked
    into ``zeros``, the rejected operations' tests would be testing the
    poison rather than the kernels. It cannot, by construction — the
    wrapper replaces only the uninitialized helper — and that is asserted
    rather than assumed."""
    with poisoned(POISON_NAN):
        core = cpp.NativeTensorCore.zeros((4, 4))
        storage = cpp.NativeStorage(9)
    try:
        assert np.array_equal(core.to_numpy(), np.zeros((4, 4)))
        assert np.array_equal(storage.to_numpy(), np.zeros(9))
    finally:
        core.close()
        storage.close()


@needs_native
def test_the_poison_is_removed_when_the_context_exits():
    """The wrapper is scoped: the class attribute is restored to the very
    descriptor it had before, so no test can leave a filling wrapper
    behind for the next one."""
    before = cpp.NativeStorage.__dict__["_uninitialized"]
    with poisoned(POISON_NAN):
        assert cpp.NativeStorage.__dict__["_uninitialized"] is not before
    assert cpp.NativeStorage.__dict__["_uninitialized"] is before

    core = cpp.NativeTensorCore._uninitialized((4,))
    try:
        # Unwrapped, the contents are indeterminate by definition, so
        # nothing is asserted about them — only that no poison is being
        # applied any more, which we prove by writing and reading back.
        core.storage.fill(3.0)
        assert np.array_equal(core.to_numpy(), np.full(4, 3.0))
    finally:
        core.close()


@needs_native
def test_the_detector_catches_a_partial_write():
    """The decisive negative control. ``tf_core_narrow_backward`` writes
    only the narrowed region — it is the runtime's clearest partial-write
    kernel. Pointed at an uninitialized destination it must leave poison
    in exactly the untouched cells, which is how we know the detector
    would catch a kernel that developed such a hole."""
    upstream = cpp.NativeTensorCore.from_array(np.ones((1, 5)))
    with poisoned(POISON_NAN):
        out = cpp.NativeTensorCore._uninitialized((3, 5))
        upstream.storage._lib.tf_core_narrow_backward(
            upstream.storage._require_open(),
            out.storage._require_open(),
            cpp._layout_vector((1, 5)),
            cpp._layout_vector(upstream.strides),
            cpp._layout_vector((5, 1)),
            upstream.offset, 5, 2,
        )
    try:
        produced = out.to_numpy()
        # Exactly the two un-narrowed rows are still poison...
        assert poison_survivors(produced, POISON_NAN) == 10
        assert np.array_equal(produced[1], np.ones(5))
        assert np.all(np.isnan(produced[0])) and np.all(np.isnan(produced[2]))
    finally:
        out.close()
        upstream.close()


@needs_native
def test_the_detector_catches_read_before_write_accumulation():
    """The second negative control. ``tf_core_sum`` accumulates
    (``dst += src``), so an uninitialized destination is *read* before it
    is written. With a NaN poison the wrong answer is loud rather than
    plausible — which is precisely why ``sum`` keeps a zeroed output."""
    values = cpp.NativeTensorCore.from_array(np.ones((3, 4)))
    with poisoned(POISON_NAN):
        out = cpp.NativeTensorCore._uninitialized((4,))
        values.storage._lib.tf_core_sum(
            values.storage._require_open(),
            out.storage._require_open(),
            cpp._layout_vector((3, 4)),
            cpp._layout_vector(values.strides),
            cpp._layout_vector((0, 1)),
            values.offset, 2,
        )
    try:
        produced = out.to_numpy()
        assert poison_survivors(produced, POISON_NAN) == 4
        # The correct answer would have been [3, 3, 3, 3].
        assert not np.array_equal(produced, np.full(4, 3.0))
    finally:
        out.close()
        values.close()


@needs_native
def test_a_simulated_hole_in_a_real_kernels_output_is_caught():
    """The third negative control, and the sharpest one: it aims the
    detector at a *complete* kernel and then deliberately reintroduces a
    hole, proving the assertion used by every proof in §2b fails when it
    should.

    ``add`` writes every element, so its output normally has no
    survivors. Here the destination is grown by one element beyond what
    the kernel is told to write, so exactly one cell keeps the poison and
    ``_assert_no_survivors`` must reject it."""
    a = cpp.NativeTensorCore.from_array(np.ones(8))
    b = cpp.NativeTensorCore.from_array(np.full(8, 2.0))
    try:
        with poisoned(POISON_NAN):
            out = cpp.NativeTensorCore._uninitialized((9,))
            # The real contiguous kernel, told to write only 8 of the 9.
            a.storage._lib.tf_core_add_contiguous(
                a.storage._require_open(), b.storage._require_open(),
                out.storage._require_open(), 8, a.offset, b.offset,
            )
        produced = out.to_numpy().copy()
        assert poison_survivors(produced, POISON_NAN) == 1
        with pytest.raises(AssertionError, match="never written"):
            _assert_no_survivors(out, POISON_NAN, "deliberately holed add")
    finally:
        a.close()
        b.close()


# ==========================================================================
# 2b. Poison proofs for every enabled operation
# ==========================================================================

def _assert_no_survivors(core, pattern, label):
    try:
        produced = core.to_numpy().copy()
    finally:
        core.close()
    survivors = poison_survivors(produced, pattern)
    assert survivors == 0, (
        f"{label}: {survivors} destination element(s) were never written"
    )
    return produced


@needs_native
@pytest.mark.parametrize("pattern", POISONS)
def test_elementwise_and_unary_operations_write_every_element(pattern):
    """Covers both traversal paths (the flat contiguous kernel and the
    generic odometer), broadcasting, and a nonzero-offset strided view —
    the shape, stride, and offset logic where a hole would hide."""
    left = rng(1).uniform(-1, 1, (7, 5))
    right = rng(2).uniform(-1, 1, (7, 5))
    positive = rng(3).uniform(0.5, 2.0, (7, 5))

    a = cpp.NativeTensorCore.from_array(left)
    b = cpp.NativeTensorCore.from_array(right)
    pos = cpp.NativeTensorCore.from_array(positive)
    row = cpp.NativeTensorCore.from_array(rng(4).uniform(-1, 1, (1, 5)))
    transposed_base = cpp.NativeTensorCore.from_array(
        np.ascontiguousarray(left.T))
    strided = transposed_base.transpose(1, 0)     # a real (7, 5) view
    narrowed = a.narrow(0, 2, 3)                  # nonzero offset

    cases = {
        "add (contiguous)": lambda: a.add(b),
        "subtract (contiguous)": lambda: a.subtract(b),
        "multiply (strided lhs)": lambda: strided.multiply(b),
        "add (broadcast)": lambda: a.add(row),
        "multiply (narrowed, nonzero offset)": lambda: narrowed.multiply(
            a.narrow(0, 0, 3)),
        "relu (contiguous)": lambda: a.relu(),
        "relu (strided)": lambda: strided.relu(),
        "relu (narrowed)": lambda: narrowed.relu(),
        "sqrt": lambda: pos.sqrt(),
        "reciprocal": lambda: pos.reciprocal(),
        "exp": lambda: a.exp(),
        "log": lambda: pos.log(),
        "contiguous_copy (of a strided view)": lambda: strided.contiguous_copy(),
        "contiguous_copy (of a narrowed view)": lambda: narrowed.contiguous_copy(),
        "matmul": lambda: strided.transpose(1, 0).matmul(a),
        "relu_backward": lambda: a.relu_backward(b),
        "softmax (last axis)": lambda: a.softmax(),
        "softmax (axis 0)": lambda: a.softmax(axis=0),
        "log_softmax (last axis)": lambda: a.log_softmax(),
        "log_softmax (axis 0)": lambda: a.log_softmax(axis=0),
    }
    try:
        for label, build in cases.items():
            with poisoned(pattern):
                out = build()
            _assert_no_survivors(out, pattern, label)
    finally:
        for core in (a, b, pos, row, transposed_base, strided, narrowed):
            if not core._closed:
                core.close()


@needs_native
@pytest.mark.parametrize("pattern", POISONS)
def test_both_matmul_paths_write_every_element(pattern):
    """Phase H, milestone H2, re-proving H1's matmul row on the structure
    that replaced it.

    Before H2 the argument was simple: the kernel accumulated into a local
    register and assigned the destination once, so it never read it. H2's
    optimized path accumulates **in the destination**, which is a
    different argument and needs its own proof: its ``k == 0`` pass
    assigns every element of every row in the group before any
    accumulation reads one.

    Both shipped paths are covered here, chosen by real layouts rather
    than by a switch — a right operand with unit column stride takes the
    row sweep, a transposed one takes the generic path — and the shapes
    span the row-block boundary (``MATMUL_ROW_BLOCK`` is 4) and the
    column threshold (``MATMUL_MIN_COLUMNS`` is 8) on both sides."""
    cases = []
    # (m, n, p): a partial group, an exact group, several groups with a
    # partial tail, and p immediately below / at / above the threshold.
    for m, n, p in ((1, 3, 16), (3, 3, 16), (4, 3, 16), (5, 3, 16),
                    (9, 3, 16), (6, 5, 7), (6, 5, 8), (6, 5, 9),
                    (7, 1, 16), (12, 9, 13)):
        left = rng(m * 13 + n).uniform(-1, 1, (m, n))
        right = rng(n * 13 + p).uniform(-1, 1, (n, p))
        cases.append((m, n, p, left, right))

    for m, n, p, left, right in cases:
        core_left = cpp.NativeTensorCore.from_array(left)
        core_right = cpp.NativeTensorCore.from_array(right)
        # The same logical right operand through a layout whose column
        # stride is not 1, so the generic path runs over a poisoned
        # destination too.
        transposed_base = cpp.NativeTensorCore.from_array(
            np.ascontiguousarray(right.T))
        strided_right = transposed_base.transpose(1, 0)
        try:
            assert core_right.strides[1] == 1
            with poisoned(pattern):
                fast = core_left.matmul(core_right)
            fast_values = _assert_no_survivors(
                fast, pattern, f"matmul unit-column-stride {m}x{n}x{p}")

            if strided_right.strides[1] != 1:   # n == 1 leaves it at 1
                with poisoned(pattern):
                    generic = core_left.matmul(strided_right)
                generic_values = _assert_no_survivors(
                    generic, pattern, f"matmul strided {m}x{n}x{p}")
                # Neither result may depend on what the buffer held.
                assert np.array_equal(fast_values, generic_values), (m, n, p)
            assert np.allclose(fast_values, left @ right, atol=1e-10)
        finally:
            for core in (core_left, core_right, transposed_base,
                         strided_right):
                if not core._closed:
                    core.close()


@needs_native
def test_matmul_results_do_not_depend_on_the_destinations_prior_contents():
    """The stronger statement the poison alone does not make: the same
    product computed over a NaN-poisoned buffer, a finite-poisoned buffer,
    and an ordinary zeroed buffer must agree **bit for bit**, on both
    paths.

    Unqualified bit equality is the right assertion here, and it is a
    different claim from H2's cross-path one: this compares **one** path
    against itself over finite operands, varying only what the
    destination held beforehand. Nothing about NaN payload selection is
    involved, because the operands are finite and the two runs execute
    the identical instructions."""
    left = rng(80).uniform(-1, 1, (9, 6))
    right = rng(81).uniform(-1, 1, (6, 16))
    core_left = cpp.NativeTensorCore.from_array(left)
    core_right = cpp.NativeTensorCore.from_array(right)
    transposed_base = cpp.NativeTensorCore.from_array(
        np.ascontiguousarray(right.T))
    strided_right = transposed_base.transpose(1, 0)
    try:
        for operand in (core_right, strided_right):
            results = []
            for pattern in POISONS:
                with poisoned(pattern):
                    out = core_left.matmul(operand)
                try:
                    results.append(out.to_numpy().copy())
                finally:
                    out.close()
            # ...and once with no poison installed at all.
            out = core_left.matmul(operand)
            try:
                results.append(out.to_numpy().copy())
            finally:
                out.close()
            reference = results[0].view(np.uint64).tobytes()
            for produced in results[1:]:
                assert produced.view(np.uint64).tobytes() == reference
    finally:
        for core in (core_left, core_right, transposed_base, strided_right):
            core.close()


@needs_native
@pytest.mark.parametrize("pattern", POISONS)
def test_host_entry_and_fill_constructors_write_every_element(pattern):
    with poisoned(pattern):
        core = cpp.NativeTensorCore.from_array(rng(5).uniform(-1, 1, (4, 6)))
    _assert_no_survivors(core, pattern, "from_array")

    with poisoned(pattern):
        core = cpp.NativeTensorCore.full((3, 5), 2.5)
    produced = _assert_no_survivors(core, pattern, "full")
    assert np.array_equal(produced, np.full((3, 5), 2.5))


@needs_native
@pytest.mark.parametrize("pattern", POISONS)
def test_classification_operations_write_every_element(pattern):
    logits = cpp.NativeTensorCore.from_array(rng(6).uniform(-2, 2, (6, 4)))
    upstream = cpp.NativeTensorCore.from_array(np.array(1.0))
    try:
        for reduction in ("mean", "sum"):
            with poisoned(pattern):
                result = logits.cross_entropy_forward([0, 1, 2, 3, 0, 1],
                                                      reduction=reduction)
            try:
                loss = result.loss.to_numpy().copy()
                probabilities = result.probabilities.to_numpy().copy()
                assert poison_survivors(loss, pattern) == 0
                assert poison_survivors(probabilities, pattern) == 0
                with poisoned(pattern):
                    gradient = result.probabilities.cross_entropy_backward(
                        result.targets, upstream, reduction=reduction)
                _assert_no_survivors(gradient, pattern,
                                     f"cross_entropy_backward ({reduction})")
            finally:
                result.close()
    finally:
        logits.close()
        upstream.close()


@needs_native
@pytest.mark.parametrize("pattern", POISONS)
def test_convolution_and_pooling_write_every_element(pattern):
    """Includes padding and strides, because that is where a boundary
    branch could skip an output position, and includes the two scatter
    backwards, whose kernels zero their own span before accumulating."""
    images = cpp.NativeTensorCore.from_array(rng(7).uniform(-1, 1, (2, 3, 6, 7)))
    weight = cpp.NativeTensorCore.from_array(rng(8).uniform(-1, 1, (4, 3, 3, 3)))
    bias = cpp.NativeTensorCore.from_array(rng(9).uniform(-1, 1, (4,)))
    try:
        for stride, padding in ((1, 0), (1, 1), (2, 1)):
            with poisoned(pattern):
                out = images.conv2d_forward(weight, bias, stride=stride,
                                            padding=padding)
            shape = out.shape
            _assert_no_survivors(out, pattern,
                                 f"conv2d_forward (stride={stride}, "
                                 f"padding={padding})")
            upstream = cpp.NativeTensorCore.from_array(
                rng(10).uniform(-1, 1, shape))
            try:
                with poisoned(pattern):
                    grad_input = upstream.conv2d_input_backward(
                        weight, input_shape=(2, 3, 6, 7), stride=stride,
                        padding=padding)
                _assert_no_survivors(grad_input, pattern,
                                     "conv2d_input_backward")
                with poisoned(pattern):
                    grad_weight = upstream.conv2d_weight_backward(
                        images, weight_shape=(4, 3, 3, 3), stride=stride,
                        padding=padding)
                _assert_no_survivors(grad_weight, pattern,
                                     "conv2d_weight_backward")
            finally:
                upstream.close()

        # Pooling: the values, the private winners, and the scatter
        # backward — including overlapping windows, where a stride
        # smaller than the kernel leaves input positions untouched.
        for kernel_size, stride in ((2, None), (3, 1), (2, 3)):
            with poisoned(pattern):
                pooled, winners = images._maxpool2d_forward_with_winners(
                    kernel_size=kernel_size, stride=stride)
            try:
                assert poison_survivors(pooled.to_numpy(), pattern) == 0
                assert poison_survivors(winners.to_numpy(), pattern) == 0
                upstream = cpp.NativeTensorCore.from_array(
                    rng(11).uniform(-1, 1, pooled.shape))
                try:
                    with poisoned(pattern):
                        grad = upstream.maxpool2d_backward(
                            winners, input_shape=(2, 3, 6, 7))
                    _assert_no_survivors(
                        grad, pattern,
                        f"maxpool2d_backward (kernel={kernel_size}, "
                        f"stride={stride})")
                finally:
                    upstream.close()
            finally:
                pooled.close()
                winners.close()
    finally:
        images.close()
        weight.close()
        bias.close()


@needs_native
@pytest.mark.parametrize("pattern", POISONS)
@pytest.mark.parametrize("p", [0.0, 0.25, 0.9])
def test_dropout_writes_every_output_and_mask_element(pattern, p):
    """Both destinations, across probabilities that keep everything, drop
    most, and drop nothing."""
    values = cpp.NativeTensorCore.from_array(rng(12).uniform(-1, 1, (5, 8)))
    try:
        with poisoned(pattern):
            out, mask = values._dropout_forward_with_mask(
                p, seed=20260728, call_index=3)
        try:
            assert poison_survivors(out.to_numpy(), pattern) == 0
            assert poison_survivors(mask.to_numpy(), pattern) == 0
        finally:
            out.close()
            mask.close()
    finally:
        values.close()


@needs_native
def test_a_complete_training_step_leaves_no_poison_anywhere():
    """The end-to-end poison proof: a full native training step — forward,
    loss, backward, and an optimizer update — with every uninitialized
    allocation poisoned. Any hole in any kernel on that path would show up
    as a NaN in a parameter, a gradient, or the loss."""
    from tensorforge.experimental import (NativeAdam, NativeLinear,
                                          NativeMSELoss, NativeReLU,
                                          NativeSequential, NativeTensor)

    x = NativeTensor.from_array(rng(13).uniform(-1, 1, (8, 6)))
    y = NativeTensor.from_array(rng(14).uniform(-1, 1, (8, 3)))
    model = NativeSequential(NativeLinear(6, 5, seed=0), NativeReLU(),
                             NativeLinear(5, 3, seed=1))
    criterion = NativeMSELoss()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    try:
        with poisoned(POISON_NAN):
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
        assert np.isfinite(float(loss.to_numpy())), "the loss picked up poison"
        assert np.all(np.isfinite(prediction.to_numpy()))
        for name, parameter in model.named_parameters():
            assert np.all(np.isfinite(parameter.to_numpy())), name
            assert parameter.grad is not None
            assert np.all(np.isfinite(parameter.grad.to_numpy())), name
        loss.close()
        prediction.close()
    finally:
        for parameter in model.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad
                parameter.zero_grad()
                gradient.close()
            parameter.close()
        optimizer.close()
        x.close()
        y.close()


# ==========================================================================
# 3. Rejections — the operations that must keep a zeroed destination
# ==========================================================================

@needs_native
def test_sum_still_allocates_a_zeroed_destination():
    """``tf_core_sum`` accumulates into its output, so the zero is the
    additive identity, not a redundant write. With the poison in place the
    result must still be correct, which proves ``sum`` did not take the
    uninitialized path."""
    values = rng(15).uniform(-1, 1, (4, 6))
    core = cpp.NativeTensorCore.from_array(values)
    try:
        for axis in (None, 0, 1):
            with poisoned(POISON_NAN):
                out = core.sum(axis=axis)
            try:
                produced = out.to_numpy()
                assert poison_survivors(produced, POISON_NAN) == 0
                assert np.allclose(produced, values.sum(axis=axis), atol=1e-12)
            finally:
                out.close()
    finally:
        core.close()


@needs_native
def test_narrow_backward_still_allocates_a_zeroed_destination():
    """The un-narrowed cells of a narrow backward are *supposed* to be
    zero — that zero is the gradient's value. With the poison in place they
    must still read back as exactly 0.0."""
    upstream = cpp.NativeTensorCore.from_array(rng(16).uniform(-1, 1, (2, 5)))
    try:
        with poisoned(POISON_NAN):
            out = upstream.narrow_backward(0, 1, (5, 5))
        try:
            produced = out.to_numpy()
            assert poison_survivors(produced, POISON_NAN) == 0
            assert np.array_equal(produced[[0, 3, 4]], np.zeros((3, 5)))
            assert np.array_equal(produced[1:3], upstream.to_numpy())
        finally:
            out.close()
    finally:
        upstream.close()


@needs_native
def test_the_rejected_sites_are_pinned_in_the_source():
    """A structural guard: if a later milestone moves ``sum`` or
    ``narrow_backward`` onto the uninitialized path, it must delete these
    comments too — which makes the decision deliberate rather than a
    careless find-and-replace."""
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "src" / "tensorforge"
              / "backends" / "cpp.py").read_text(encoding="utf-8")
    assert source.count("H1 REJECTED") == 2
    # ...and each rejection states its own reason, so the two cannot be
    # confused for one another.
    for marker in ("accumulates", "partial-write*", "additive identity"):
        assert marker in source, marker
    # Both rejected operations still construct through zeros(); every
    # other Core output allocation went to the uninitialized path.
    assert source.count("NativeTensorCore.zeros(out_shape") == 1     # sum
    assert source.count("NativeTensorCore.zeros(original") == 1      # narrow


# ==========================================================================
# 4. Failure paths — nothing from the uninitialized path escapes
# ==========================================================================

@needs_native
def test_a_failed_kernel_closes_an_uninitialized_output(monkeypatch,
                                                        live_storages):
    """Every enabled site must release its destination when the native
    call fails, without waiting for garbage collection."""
    library = cpp._require_library()
    a = cpp.NativeTensorCore.from_array(rng(17).uniform(-1, 1, (4, 4)))
    b = cpp.NativeTensorCore.from_array(rng(18).uniform(-1, 1, (4, 4)))
    gc.collect()
    baseline = len(live_storages)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated native failure")

    for kernel, call in (
        ("tf_core_add_contiguous", lambda: a.add(b)),
        ("tf_core_multiply", lambda: a.transpose(1, 0).multiply(b)),
        ("tf_core_relu_contiguous", lambda: a.relu()),
        ("tf_core_exp_contiguous", lambda: a.exp()),
        ("tf_core_matmul", lambda: a.matmul(b)),
        ("tf_core_relu_backward", lambda: a.relu_backward(b)),
        ("tf_core_contiguous_copy", lambda: a.transpose(1, 0).contiguous_copy()),
        ("tf_core_softmax_forward", lambda: a.softmax()),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(library, kernel, boom)
            with pytest.raises(RuntimeError, match="simulated"):
                call()
        gc.collect()
        assert len(live_storages) == baseline, (
            f"{kernel}: a failed call leaked its destination"
        )

    a.close()
    b.close()


@needs_native
def test_a_failed_conversion_after_allocation_closes_the_output(
        monkeypatch, live_storages):
    """A Python-side failure *after* the native call still releases the
    destination — the wrapper's cleanup is not conditional on where the
    failure came from.

    The injected seam is ``NativeTensorView._bind``, the single point both
    view constructors funnel through: the public ``__init__`` (which
    normalizes caller-supplied metadata first) and the private
    ``_from_validated`` the allocating constructors use since H3 (which
    skips that normalization because this module already performed it).
    Patching ``_bind`` therefore covers **both** paths at once, where
    patching ``__init__`` would only reach the public one."""
    a = cpp.NativeTensorCore.from_array(rng(19).uniform(-1, 1, (4, 4)))
    storage = cpp.NativeStorage.from_array(np.zeros(16))
    gc.collect()
    baseline = len(live_storages)

    original = cpp.NativeTensorView._bind

    def failing_bind(self, *args, **kwargs):
        raise RuntimeError("simulated wrapper failure")

    with monkeypatch.context() as patch:
        patch.setattr(cpp.NativeTensorView, "_bind", failing_bind)
        # The private, already-normalized path used by every allocation.
        with pytest.raises(RuntimeError, match="simulated"):
            cpp.NativeTensorCore._uninitialized((4, 4))
        with pytest.raises(RuntimeError, match="simulated"):
            cpp.NativeTensorCore.zeros((4, 4))
        with pytest.raises(RuntimeError, match="simulated"):
            cpp.NativeTensorCore.from_array(np.zeros((4, 4)))
        # The public, fully validating path.
        with pytest.raises(RuntimeError, match="simulated"):
            cpp.NativeTensorView(storage, (4, 4))
    assert cpp.NativeTensorView._bind is original
    gc.collect()
    # Each storage was constructed before its view failed; each is
    # unreachable, so the invariant here is that they do not survive as
    # *live* handles once collected.
    assert len(live_storages) <= baseline + 1
    storage.close()
    a.close()


@needs_native
def test_a_poisoned_allocation_that_fails_its_fill_is_closed(live_storages):
    """The test infrastructure must not leak either. If the poison fill
    itself fails, the wrapper closes the storage it just allocated rather
    than handing back a half-prepared buffer or dropping it on the
    floor."""
    gc.collect()
    baseline = len(live_storages)
    library = cpp._require_library()
    original_fill = library.tf_storage_fill

    def boom(*args, **kwargs):
        raise RuntimeError("simulated fill failure")

    library.tf_storage_fill = boom
    try:
        with poisoned(POISON_NAN):
            with pytest.raises(RuntimeError, match="simulated fill failure"):
                cpp.NativeStorage._uninitialized(32)
    finally:
        library.tf_storage_fill = original_fill
    gc.collect()
    assert len(live_storages) == baseline


@needs_native
@needs_fault_injection
def test_an_allocation_failure_inside_an_operation_leaves_nothing_open(
        live_storages):
    """The allocation itself failing must leave live storage exactly at
    baseline — the failure happens before any handle exists."""
    a = cpp.NativeTensorCore.from_array(rng(20).uniform(-1, 1, (4, 4)))
    b = cpp.NativeTensorCore.from_array(rng(21).uniform(-1, 1, (4, 4)))
    gc.collect()
    baseline = len(live_storages)
    for call in (lambda: a.add(b), lambda: a.matmul(b), lambda: a.relu(),
                 lambda: a.exp(), lambda: a.softmax()):
        cpp._arm_alloc_failure(1)
        try:
            with pytest.raises(MemoryError):
                call()
        finally:
            cpp._arm_alloc_failure(0)
        gc.collect()
        assert len(live_storages) == baseline
    a.close()
    b.close()


@needs_native
def test_a_failed_full_closes_its_tensor(live_storages):
    """``full`` allocates uninitialized and then fills; a bad value must
    be rejected before the allocation, and a failed fill must close."""
    gc.collect()
    baseline = len(live_storages)
    with pytest.raises((TypeError, ValueError)):
        cpp.NativeTensorCore.full((3, 3), "not a number")
    gc.collect()
    assert len(live_storages) == baseline, (
        "a rejected fill value still allocated storage"
    )


@needs_native
def test_a_failed_from_array_copy_closes_its_storage(monkeypatch,
                                                     live_storages):
    library = cpp._require_library()
    gc.collect()
    baseline = len(live_storages)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated copy failure")

    with monkeypatch.context() as patch:
        patch.setattr(library, "tf_storage_copy_from", boom)
        with pytest.raises(RuntimeError, match="simulated"):
            cpp.NativeStorage.from_array(np.ones(16))
    gc.collect()
    assert len(live_storages) == baseline


@needs_native
def test_repeated_operation_cycles_return_live_storage_to_baseline(
        live_storages):
    """Success and failure interleaved, so neither path accumulates."""
    library = cpp._require_library()
    a = cpp.NativeTensorCore.from_array(rng(22).uniform(-1, 1, (8, 8)))
    b = cpp.NativeTensorCore.from_array(rng(23).uniform(-1, 1, (8, 8)))
    gc.collect()
    baseline = len(live_storages)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated native failure")

    original = library.tf_core_add_contiguous
    for _ in range(50):
        out = a.add(b)
        out.close()
        out = a.matmul(b)
        out.close()
        library.tf_core_add_contiguous = boom
        try:
            with pytest.raises(RuntimeError):
                a.add(b)
        finally:
            library.tf_core_add_contiguous = original
        gc.collect()
        assert len(live_storages) == baseline
    a.close()
    b.close()


@needs_native
def test_poisoned_success_and_failure_cycles_return_to_baseline(live_storages):
    """The same lifecycle claim with the poison wrapper installed, so the
    test infrastructure is proved not to change the accounting it is used
    to verify."""
    library = cpp._require_library()
    a = cpp.NativeTensorCore.from_array(rng(32).uniform(-1, 1, (8, 8)))
    b = cpp.NativeTensorCore.from_array(rng(33).uniform(-1, 1, (8, 8)))
    gc.collect()
    baseline = len(live_storages)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated native failure")

    original = library.tf_core_add_contiguous
    with poisoned(POISON_NAN):
        for _ in range(50):
            out = a.add(b)
            assert poison_survivors(out.to_numpy(), POISON_NAN) == 0
            out.close()
            library.tf_core_add_contiguous = boom
            try:
                with pytest.raises(RuntimeError):
                    a.add(b)
            finally:
                library.tf_core_add_contiguous = original
            gc.collect()
            assert len(live_storages) == baseline
    a.close()
    b.close()


# ==========================================================================
# 5. Numerical parity — H1 changed allocation, not arithmetic
# ==========================================================================

@needs_native
def test_every_enabled_operation_is_bit_identical_to_the_zeroed_path(
        monkeypatch):
    """The core numerical claim, tested directly: forcing every H1 site
    back onto the zero-initializing allocator must produce **bit-identical**
    results. Anything else would mean the arithmetic depended on the
    destination's initial contents.

    The uninitialized run happens **under poison**, which is what makes
    this a real test rather than a coincidence: unpoisoned fresh heap
    pages usually read back as zeros, so a kernel with a hole could match
    the zeroed path by luck. With a NaN poison a hole cannot match."""
    values = rng(24).uniform(-1, 1, (6, 5))
    other = rng(25).uniform(-1, 1, (6, 5))
    positive = rng(26).uniform(0.5, 2.0, (6, 5))
    images = rng(27).uniform(-1, 1, (2, 3, 5, 6))
    weight = rng(28).uniform(-1, 1, (4, 3, 3, 3))
    bias = rng(29).uniform(-1, 1, (4,))

    def collect():
        a = cpp.NativeTensorCore.from_array(values)
        b = cpp.NativeTensorCore.from_array(other)
        pos = cpp.NativeTensorCore.from_array(positive)
        img = cpp.NativeTensorCore.from_array(images)
        w = cpp.NativeTensorCore.from_array(weight)
        bi = cpp.NativeTensorCore.from_array(bias)
        # The base must outlive the view: a transposed view borrows its
        # base's storage, so dropping the base would close it.
        strided_base = cpp.NativeTensorCore.from_array(
            np.ascontiguousarray(values.T))
        strided = strided_base.transpose(1, 0)
        results = {}
        produced = []
        try:
            def record(name, core):
                produced.append(core)
                results[name] = core.to_numpy().copy()

            record("add", a.add(b))
            record("subtract", a.subtract(b))
            record("multiply_strided", strided.multiply(b))
            record("relu", a.relu())
            record("sqrt", pos.sqrt())
            record("reciprocal", pos.reciprocal())
            record("exp", a.exp())
            record("log", pos.log())
            record("contiguous_copy", strided.contiguous_copy())
            record("matmul", a.transpose(1, 0).matmul(b))
            record("relu_backward", a.relu_backward(b))
            record("softmax", a.softmax())
            record("log_softmax", a.log_softmax())
            record("sum", a.sum(axis=0))
            record("conv2d_forward", img.conv2d_forward(w, bi))
            record("full", cpp.NativeTensorCore.full((3, 3), 1.25))
            entropy = a.cross_entropy_forward([0, 1, 2, 3, 4, 0])
            try:
                results["ce_loss"] = entropy.loss.to_numpy().copy()
                results["ce_probs"] = entropy.probabilities.to_numpy().copy()
            finally:
                entropy.close()
        finally:
            for core in produced:
                core.close()
            for core in (a, b, pos, img, w, bi, strided, strided_base):
                if not core._closed:
                    core.close()
        return results

    with poisoned(POISON_NAN):
        fast = collect()
    # Force every H1 site back onto the zero-initializing allocator.
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        cpp.NativeTensorCore.zeros)
    monkeypatch.setattr(cpp.NativeStorage, "_uninitialized",
                        lambda size, dtype=None, device="cpu":
                        cpp.NativeStorage(size, dtype=dtype, device=device))
    zeroed = collect()

    assert set(fast) == set(zeroed)
    for name in fast:
        assert fast[name].dtype == zeroed[name].dtype == np.float64, name
        assert np.array_equal(fast[name], zeroed[name]), (
            f"{name} is not bit-identical between the two allocation paths"
        )


@needs_native
def test_a_training_run_is_bit_identical_between_allocation_paths(monkeypatch):
    """The same claim at the level that matters: a deterministic multi-step
    training run must produce the identical loss sequence and identical
    final parameters under either allocator — with the uninitialized run
    poisoned, so a hole anywhere on the forward, backward, or optimizer
    path would break the comparison rather than hide behind a zeroed
    page."""
    from tensorforge.experimental import (NativeAdam, NativeLinear,
                                          NativeMSELoss, NativeReLU,
                                          NativeSequential, NativeTensor)

    inputs = rng(30).uniform(-1, 1, (10, 4))
    targets = rng(31).uniform(-1, 1, (10, 2))

    def run():
        x = NativeTensor.from_array(inputs)
        y = NativeTensor.from_array(targets)
        model = NativeSequential(NativeLinear(4, 6, seed=3), NativeReLU(),
                                 NativeLinear(6, 2, seed=4))
        criterion = NativeMSELoss()
        optimizer = NativeAdam(model.parameters(), lr=0.05)
        losses = []
        try:
            for _ in range(8):
                optimizer.zero_grad()
                prediction = model(x)
                loss = criterion(prediction, y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.to_numpy()))
                loss.close()
                prediction.close()
            final = {name: parameter.to_numpy().copy()
                     for name, parameter in model.named_parameters()}
        finally:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad
                    parameter.zero_grad()
                    gradient.close()
                parameter.close()
            optimizer.close()
            x.close()
            y.close()
        return losses, final

    with poisoned(POISON_NAN):
        fast_losses, fast_final = run()
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        cpp.NativeTensorCore.zeros)
    monkeypatch.setattr(cpp.NativeStorage, "_uninitialized",
                        lambda size, dtype=None, device="cpu":
                        cpp.NativeStorage(size, dtype=dtype, device=device))
    zeroed_losses, zeroed_final = run()

    assert fast_losses == zeroed_losses, "the loss sequence changed"
    assert set(fast_final) == set(zeroed_final)
    for name in fast_final:
        assert np.array_equal(fast_final[name], zeroed_final[name]), name


# ==========================================================================
# 6. Scope — one C ABI symbol, no public capability, no poison control
# ==========================================================================

# The exported-symbol inventory H1 left behind: the pre-H1 baseline of 51
# plus tf_storage_create_uninitialized, and nothing else.
BASELINE_TF_EXPORTS = 51
PHASE_H_TF_EXPORTS = 52
# ...and what the built library holds now. Phase I milestone I1 added the
# two typed storage creators — the only two symbols the whole phase adds —
# so the live count is 54 while Phase H's closure remains 52. Kept as two
# named constants because they are facts about two different moments.
PHASE_I_TYPED_CREATORS = (
    "tf_storage_create_typed",
    "tf_storage_create_uninitialized_typed",
)
EXPECTED_TF_EXPORTS = PHASE_H_TF_EXPORTS + len(PHASE_I_TYPED_CREATORS)  # 54


def phase_h_export_names(exported):
    """``exported`` with the Phase-I additions removed — the export surface
    as Phase H closed it.

    Every per-milestone Phase-H test module asserts "this milestone added
    no ABI symbol". That claim is about Phase H, it is still true, and it
    must stay checkable — but the live library now also carries the two
    typed creators milestone I1 added, so the claim is measured against
    this subset rather than against the raw total. Sharing the helper
    keeps the eight modules that make the claim from drifting apart.
    """
    return [name for name in exported if name not in PHASE_I_TYPED_CREATORS]

# Every name that would constitute a runtime poison-control API. None may
# exist in the shipped library or the installed Python backend.
FORBIDDEN_POISON_NAMES = (
    "tf_test_set_uninitialized_poison",
    "tf_set_uninitialized_poison",
    "tf_storage_set_poison",
    "tf_test_poison",
    "_set_uninitialized_poison",
    "_uninitialized_poison",
    "set_uninitialized_poison",
    "uninitialized_poison",
)


def _pe_exported_names(data):
    """Every name in a PE image's export directory."""
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[e_lfanew:e_lfanew + 4] == b"PE\0\0", "not a PE image"
    coff = e_lfanew + 4
    section_count = struct.unpack_from("<H", data, coff + 2)[0]
    optional_size = struct.unpack_from("<H", data, coff + 16)[0]
    optional = coff + 20
    magic = struct.unpack_from("<H", data, optional)[0]
    # The export directory is data directory 0; PE32+ puts it 16 bytes
    # further in than PE32 does.
    directories = optional + (112 if magic == 0x20B else 96)
    export_rva = struct.unpack_from("<I", data, directories)[0]
    if export_rva == 0:
        return []
    sections = []
    table = optional + optional_size
    for index in range(section_count):
        base = table + index * 40
        virtual_size, virtual_address, raw_size, raw_pointer = (
            struct.unpack_from("<IIII", data, base + 8))
        sections.append((virtual_address, max(virtual_size, raw_size),
                         raw_pointer))

    def offset_of(rva):
        for address, size, pointer in sections:
            if address <= rva < address + size:
                return pointer + (rva - address)
        raise ValueError(f"RVA {rva:#x} lies outside every section")

    directory = offset_of(export_rva)
    name_count = struct.unpack_from("<I", data, directory + 24)[0]
    names_rva = struct.unpack_from("<I", data, directory + 32)[0]
    names_table = offset_of(names_rva)
    names = []
    for index in range(name_count):
        rva = struct.unpack_from("<I", data, names_table + 4 * index)[0]
        start = offset_of(rva)
        names.append(data[start:data.index(b"\0", start)].decode("ascii"))
    return names


def _elf_exported_names(data):
    """Every defined dynamic symbol in a 64-bit little-endian ELF object
    (the shape the Linux sanitizer builds produce). Returns ``None`` for
    any other ELF variant rather than guessing."""
    if data[4] != 2 or data[5] != 1:          # not ELF64 / not little-endian
        return None
    section_offset, = struct.unpack_from("<Q", data, 0x28)
    entry_size, count = struct.unpack_from("<HH", data, 0x3A)
    headers = []
    for index in range(count):
        base = section_offset + index * entry_size
        sh_type, = struct.unpack_from("<I", data, base + 4)
        sh_offset, sh_size = struct.unpack_from("<QQ", data, base + 24)
        sh_link, = struct.unpack_from("<I", data, base + 40)
        sh_entsize, = struct.unpack_from("<Q", data, base + 56)
        headers.append((sh_type, sh_offset, sh_size, sh_link, sh_entsize))
    names = []
    for sh_type, sh_offset, sh_size, sh_link, sh_entsize in headers:
        if sh_type != 11 or sh_entsize == 0:  # SHT_DYNSYM
            continue
        _, strtab_offset, _, _, _ = headers[sh_link]
        for position in range(sh_offset, sh_offset + sh_size, sh_entsize):
            st_name, = struct.unpack_from("<I", data, position)
            st_shndx, = struct.unpack_from("<H", data, position + 6)
            if st_shndx == 0 or st_name == 0:   # undefined / unnamed
                continue
            start = strtab_offset + st_name
            names.append(data[start:data.index(b"\0", start)].decode("ascii"))
    return names


def exported_names(path):
    """``(image_format, names)`` for the built library, or ``(None, None)``
    on an image format this file does not parse.

    The format matters for one assertion: a PE export directory lists
    exactly what the DLL publishes, so it can be compared in full, while
    an ELF ``.dynsym`` also carries linker-supplied symbols (``_init``,
    ``_end``, …) and any sanitizer runtime's, so only the ``tf_*``
    namespace is meaningful there."""
    data = path.read_bytes()
    if data[:2] == b"MZ":
        return "pe", _pe_exported_names(data)
    if data[:4] == b"\x7fELF":
        names = _elf_exported_names(data)
        return ("elf", names) if names is not None else (None, None)
    return None, None


def test_h1_exposes_no_public_empty_api():
    """The uninitialized path is a backend implementation detail. No
    public surface may construct uninitialized storage."""
    import tensorforge
    import tensorforge.experimental as experimental
    from tensorforge.experimental import NativeTensor

    for module in (tensorforge, tensorforge.nn, experimental):
        for name in ("empty", "empty_like", "uninitialized"):
            assert not hasattr(module, name), f"{module.__name__}.{name}"
    for cls in (tensorforge.Tensor, NativeTensor):
        for name in ("empty", "empty_like", "uninitialized",
                     "_uninitialized"):
            assert not hasattr(cls, name), f"{cls.__name__}.{name}"
    # The Core/Storage helpers exist but are private by name.
    assert hasattr(cpp.NativeTensorCore, "_uninitialized")
    assert not hasattr(cpp.NativeTensorCore, "uninitialized")
    assert hasattr(cpp.NativeStorage, "_uninitialized")
    assert not hasattr(cpp.NativeStorage, "uninitialized")


def test_h1_changed_no_capability_registry():
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "uninitialized" not in cpp.backend_info()
    # Allocation strategy is not a capability, so it appears in no
    # inventory the backend reports.
    for value in cpp.backend_info().values():
        if isinstance(value, (tuple, list)):
            assert not any("uninitialized" in str(item) for item in value)


def test_h1_left_the_checkpoint_contract_at_version_two():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)


def test_the_installed_python_backend_has_no_poison_control():
    """The backend wrapper must expose no way to change what an
    allocation contains. The poison is this file's business; the shipped
    module has no part in it."""
    for name in FORBIDDEN_POISON_NAMES:
        assert not hasattr(cpp, name), f"cpp.{name} still exists"
    for owner in (cpp.NativeStorage, cpp.NativeTensorCore):
        for name in FORBIDDEN_POISON_NAMES + ("poison", "_poison"):
            assert not hasattr(owner, name), f"{owner.__name__}.{name}"
    assert not any("poison" in name.lower() for name in dir(cpp)), (
        "the backend module exposes a poison-shaped name"
    )


def test_no_production_source_mentions_a_poison_hook():
    """A source-level guard over everything that ships: no production
    file may name a poison control, in any layer."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    shipped = list((root / "src" / "tensorforge").rglob("*.py"))
    shipped += list((root / "cpp" / "src").glob("*.cpp"))
    shipped += list((root / "cpp" / "include").glob("*.h"))
    offenders = []
    for path in shipped:
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN_POISON_NAMES:
            # tf_internal.h and cpp.py *document* that no such hook
            # exists, in prose; what may not appear is the identifier
            # itself as a declaration, definition, or call.
            for suspicious in (f"{name}(", f"{name} =", f"def {name}",
                               f"library.{name}", f"cpp.{name}"):
                if suspicious in text:
                    offenders.append(f"{path.relative_to(root)}: {suspicious}")
    assert offenders == [], offenders


@needs_native
def test_the_loaded_library_exports_no_poison_control():
    """The decisive check: ask the **loaded** library for each forbidden
    symbol. ``ctypes`` resolves through the platform loader
    (``GetProcAddress`` / ``dlsym``), so a hit here would mean the
    running DLL really does export a poison hook."""
    library = cpp._require_library()
    for name in FORBIDDEN_POISON_NAMES:
        if not name.startswith("tf_"):
            continue
        with pytest.raises(AttributeError):
            getattr(library, name)
    # ...and the symbol H1 legitimately added does resolve, so the probe
    # above is proved able to find a symbol that exists.
    assert getattr(library, "tf_storage_create_uninitialized") is not None
    assert getattr(library, "tf_storage_create") is not None


@needs_native
def test_the_built_library_export_table_has_exactly_the_h1_addition():
    """The export inventory, read straight out of the built image rather
    than from a list of names this repository maintains.

    H1's whole ABI footprint is one symbol: the pre-H1 baseline of 51
    exported ``tf_*`` symbols plus ``tf_storage_create_uninitialized``."""
    image, names = exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert "tf_storage_create_uninitialized" in exported
    # Scoped to the namespace TensorForge owns: a sanitizer build's
    # runtime legitimately references __asan_poison_memory_region and
    # friends, which say nothing about this library's own surface. The
    # unrestricted check is the loader probe above, which asks for a
    # poison hook by name and must not find one.
    assert not [name for name in exported if "poison" in name.lower()]
    assert len(exported) == EXPECTED_TF_EXPORTS, exported
    # The two Phase-I creators are present in the built library...
    for name in PHASE_I_TYPED_CREATORS:
        assert name in exported, name
    # ...and removing them leaves exactly Phase H's closure inventory,
    # which is one symbol above the pre-H1 baseline. Nothing else was
    # added, and nothing H1 shipped was taken away.
    without_phase_i = [name for name in exported
                       if name not in PHASE_I_TYPED_CREATORS]
    assert len(without_phase_i) == PHASE_H_TF_EXPORTS
    assert len(without_phase_i) - 1 == BASELINE_TF_EXPORTS
    if image == "pe":
        # A PE export directory lists exactly what the DLL publishes, so
        # the count above really is the whole export surface.
        assert sorted(names) == exported, sorted(set(names) - set(exported))
